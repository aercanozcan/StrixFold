#!/usr/bin/env python3
"""GPU-free HTTP regression of the image routes with the real DeepSeek-V4.1 tokenizer and template.

fake_serve (zig/tests/server/build.sh -> zig-out/server/fake_serve) with TF_FAKE_VISION=1 takes an image as four rows,
or eight when its payload is "image-B" (so the order of two images shows in their spans), so this checks the server's request paths (chat, Messages, tool results, /tokenize, count_tokens), not pixels or the
tower. Cases after upstream PR #6's test_vision_http.py, plus the Anthropic routes.

    TF_VISION_TEST_TOKENIZER=<dir with tokenizer.json + chat_template.jinja> python3 -m unittest tools/dsv41-hip/test_vision_http.py
TF_VISION_TEST_SERVER selects another fake_serve binary.
"""
import base64
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request

SRC = Path(__file__).resolve().parents[2]
IMAGE_TOKEN = 129264
PNG = 'data:image/png;base64,aA=='
TOOLS = [{'type': 'function', 'function': {'name': 'screenshot', 'parameters': {'type': 'object', 'properties': {}}}}]


def image_url(payload):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(payload.encode()).decode()}}


def spans(tokens):
    return [len(list(group)) for token, group in itertools.groupby(tokens) if token == IMAGE_TOKEN]


@unittest.skipUnless(os.environ.get('TF_VISION_TEST_TOKENIZER'), 'set TF_VISION_TEST_TOKENIZER to the model tokenizer directory')
class ImageRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        log_path = Path(cls.temp.name) / 'server.log'
        cls.log = log_path.open('w')
        cls.addClassCleanup(cls.log.close)
        env = dict(os.environ, TF_FAKE_VISION='1', TF_FAKE_VISION_MAX='16', TF_FAKE_TOKENIZER=os.environ['TF_VISION_TEST_TOKENIZER'])
        server = os.environ.get('TF_VISION_TEST_SERVER', str(SRC / 'zig-out/server/fake_serve'))
        cls.process = subprocess.Popen([server, 'serve', str(SRC / 'zig/tests/server/fixtures'), '--host', '127.0.0.1',
                                        '--port', '0', '--name', 'dsv41'], env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(cls.close_server)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            match = re.search(r'PORT (\d+)', log_path.read_text())
            if match:
                cls.url = 'http://127.0.0.1:' + match[1]
                return
            if cls.process.poll() is not None:
                raise RuntimeError(log_path.read_text())
            time.sleep(.02)
        raise TimeoutError('fake_serve did not announce a port')

    @classmethod
    def close_server(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait()

    def post(self, route, body):
        request = urllib.request.Request(self.url + route, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode()
                return response.status, raw if body.get('stream') else json.loads(raw)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def chat(self, count=1, text='What is shown?'):
        parts = [{'type': 'text', 'text': text}] + [{'type': 'image_url', 'image_url': {'url': PNG}} for _ in range(count)]
        return {'model': 'dsv41', 'messages': [{'role': 'user', 'content': parts}], 'max_tokens': 16, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False}}

    @staticmethod
    def anthropic_image():
        return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'aA=='}}

    def test_chat_and_tokenize_agree_on_image_spans(self):
        for count in (1, 2):
            body = self.chat(count)
            status, answer = self.post('/v1/chat/completions', body)
            self.assertEqual(status, 200, answer)
            status, tokens = self.post('/v1/tokenize', body)
            self.assertEqual(status, 200, tokens)
            self.assertEqual(tokens['count'], answer['usage']['prompt_tokens'])
            self.assertEqual(tokens['tokens'].count(IMAGE_TOKEN), count * 4)

    def test_literal_placeholder_with_image_creates_no_extra_span(self):
        status, result = self.post('/v1/tokenize', self.chat(text='Literal <｜deepseek_image｜>'))
        self.assertEqual(status, 200, result)
        self.assertEqual(result['tokens'].count(IMAGE_TOKEN), 4)

    def test_text_only_literal_placeholder_remains_refused(self):
        body = self.chat(0)
        body['messages'][0]['content'] = 'Literal <｜deepseek_image｜>'
        status, _ = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 400)

    def test_remote_image_urls_remain_refused(self):
        body = self.chat()
        body['messages'][0]['content'][1]['image_url']['url'] = 'http://127.0.0.1/private'
        status, _ = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 400)

    def test_streamed_image_reply_finishes(self):
        body = dict(self.chat(), stream=True, stream_options={'include_usage': True})
        status, result = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 200)
        self.assertIn('data: [DONE]', result)

    def test_messages_image_block_and_count_tokens(self):
        body = {'model': 'dsv41', 'max_tokens': 16, 'temperature': 0, 'thinking': {'type': 'disabled'},
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is shown?'}, self.anthropic_image()]}]}
        status, answer = self.post('/v1/messages', body)
        self.assertEqual(status, 200, answer)
        status, counted = self.post('/v1/messages/count_tokens', body)
        self.assertEqual(status, 200, counted)
        self.assertEqual(counted['input_tokens'], answer['usage']['input_tokens'])

    def test_tool_result_with_image(self):
        tools = [{'name': 'screenshot', 'description': 'Take a screenshot', 'input_schema': {'type': 'object', 'properties': {}}}]
        body = {'model': 'dsv41', 'max_tokens': 16, 'temperature': 0, 'thinking': {'type': 'disabled'}, 'tools': tools,
                'messages': [
                    {'role': 'user', 'content': 'Take a screenshot and describe it.'},
                    {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_1', 'name': 'screenshot', 'input': {}}]},
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1',
                                                  'content': [{'type': 'text', 'text': 'here'}, self.anthropic_image()]}]}]}
        status, answer = self.post('/v1/messages', body)
        self.assertEqual(status, 200, answer)
        status, counted = self.post('/v1/messages/count_tokens', body)
        self.assertEqual(status, 200, counted)
        self.assertEqual(counted['input_tokens'], answer['usage']['input_tokens'])

    def test_openai_tool_message_with_image(self):
        body = self.chat()
        body['messages'] = [
            {'role': 'user', 'content': 'Take a screenshot.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function',
                                                                  'function': {'name': 'screenshot', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': [{'type': 'image_url', 'image_url': {'url': PNG}}]}]
        body['tools'] = [{'type': 'function', 'function': {'name': 'screenshot', 'parameters': {'type': 'object', 'properties': {}}}}]
        status, answer = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 200, answer)
        status, tokens = self.post('/v1/tokenize', body)
        self.assertEqual(status, 200, tokens)
        self.assertEqual(tokens['tokens'].count(IMAGE_TOKEN), 4)


    def test_user_image_and_tool_result_image_keep_their_order(self):
        for user, tool, expected in (('image-A', 'image-B', [4, 8]), ('image-B', 'image-A', [8, 4])):
            body = self.chat(0)
            body['tools'] = TOOLS
            body['messages'] = [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'Compare this with a screenshot.'}, image_url(user)]},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function',
                                                                      'function': {'name': 'screenshot', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': [{'type': 'text', 'text': 'here'}, image_url(tool)]}]
            status, tokens = self.post('/v1/tokenize', body)
            self.assertEqual(status, 200, tokens)
            self.assertEqual(spans(tokens['tokens']), expected)
            status, answer = self.post('/v1/chat/completions', body)
            self.assertEqual(status, 200, answer)

    def test_reordered_tool_results_keep_each_image(self):
        # after upstream PR #6's test_tool_result_reordering_preserves_image_identity
        def result(identity, payload):
            return {'role': 'tool', 'tool_call_id': identity, 'content': [{'type': 'text', 'text': 'Screenshot ' + identity},
                                                                         image_url(payload)]}
        calls = [{'id': i, 'type': 'function', 'function': {'name': 'screenshot', 'arguments': '{}'}} for i in 'ab']
        head = [{'role': 'user', 'content': 'Inspect both screenshots.'}, {'role': 'assistant', 'content': '', 'tool_calls': calls}]
        body = self.chat(0)
        body['tools'] = TOOLS
        seen = []
        for order in ((result('a', 'image-A'), result('b', 'image-B')), (result('b', 'image-B'), result('a', 'image-A'))):
            body['messages'] = head + list(order)
            status, tokens = self.post('/v1/tokenize', body)
            self.assertEqual(status, 200, tokens)
            self.assertEqual(spans(tokens['tokens']), [4, 8])  # the template puts the results in the calls' order
            seen.append(tokens['tokens'])
        self.assertEqual(seen[0], seen[1])

    def test_images_over_the_engine_cap_are_refused_with_400(self):
        body = self.chat(0)
        body['messages'][0]['content'] = [{'type': 'text', 'text': 'Compare.'}] + [image_url('image-B')] * 2
        status, answer = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 200, answer)  # 16 image tokens: at the cap
        body['messages'][0]['content'].append(image_url('image-B'))
        status, answer = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 400, answer)  # 24: over it
        self.assertIn('image tokens', answer['error']['message'])

if __name__ == '__main__':
    unittest.main()
