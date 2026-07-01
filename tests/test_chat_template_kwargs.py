"""
채팅 템플릿 kwargs 전달 기능 테스트

이 테스트는 클라이언트가 chat_template_kwargs를 통해 enable_thinking 등의
템플릿 변수를 제어할 수 있는지 검증합니다.
"""

import pytest

from vllm_mlx.api.models import ChatCompletionRequest


class TestChatTemplateKwargsRequest:
    """ChatCompletionRequest의 chat_template_kwargs 필드 테스트"""

    def test_request_with_chat_template_kwargs(self):
        """chat_template_kwargs가 정상적으로 파싱되는지 검증"""
        # Given
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # When
        request = ChatCompletionRequest(**request_data)

        # Then
        assert request.chat_template_kwargs is not None
        assert request.chat_template_kwargs == {"enable_thinking": False}

    def test_request_without_chat_template_kwargs(self):
        """chat_template_kwargs를 보내지 않을 때 기본값 None 검증"""
        # Given
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # When
        request = ChatCompletionRequest(**request_data)

        # Then
        assert request.chat_template_kwargs is None

    def test_request_with_multiple_kwargs(self):
        """여러 키가 포함된 chat_template_kwargs 파싱 검증"""
        # Given
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "custom_key": "custom_value",
                "another_key": 123,
            },
        }

        # When
        request = ChatCompletionRequest(**request_data)

        # Then
        assert request.chat_template_kwargs is not None
        assert request.chat_template_kwargs["enable_thinking"] is True
        assert request.chat_template_kwargs["custom_key"] == "custom_value"
        assert request.chat_template_kwargs["another_key"] == 123

    def test_request_with_empty_chat_template_kwargs(self):
        """빈 chat_template_kwargs 딕셔너리 처리 검증"""
        # Given
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {},
        }

        # When
        request = ChatCompletionRequest(**request_data)

        # Then
        assert request.chat_template_kwargs == {}

    def test_request_with_enable_thinking_false(self):
        """enable_thinking=False가 명시적으로 전달되는지 검증"""
        # Given
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # When
        request = ChatCompletionRequest(**request_data)

        # Then
        assert request.chat_template_kwargs is not None
        assert request.chat_template_kwargs["enable_thinking"] is False

    def test_request_with_enable_thinking_true(self):
        """enable_thinking=True가 명시적으로 전달되는지 검증"""
        # Given
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }

        # When
        request = ChatCompletionRequest(**request_data)

        # Then
        assert request.chat_template_kwargs is not None
        assert request.chat_template_kwargs["enable_thinking"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
