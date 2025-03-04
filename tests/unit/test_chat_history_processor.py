import pytest
from typing import List, Dict, Any
from semantic_kernel.contents.chat_history import ChatHistory
from promptflow_tool_semantic_kernel.tools.chat_history_processor import ChatHistoryProcessor


def test_build_history_with_empty_list():
    # Test with an empty chat history
    result = ChatHistoryProcessor.build_history([], "Hello")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Hello"


def test_build_history_with_dict_entries():
    # Test with dictionary entries
    chat_history = [{
        "inputs": {
            "question": "What is Python?"
        },
        "outputs": {
            "answer": {
                "content": "Python is a programming language."
            }
        }
    }, {
        "inputs": {
            "question": "Is it easy to learn?"
        },
        "outputs": {
            "answer": {
                "content": "Yes, Python is relatively easy to learn."
            }
        }
    }]

    result = ChatHistoryProcessor.build_history(chat_history, "Tell me more")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 5
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "What is Python?"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "Python is a programming language."
    assert result.messages[4].content == "Tell me more"


def test_build_history_with_string_entries():
    # Test with string entries
    chat_history = ["Hello there", "How are you?"]

    result = ChatHistoryProcessor.build_history(chat_history,
                                                "I'm fine, thanks")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 3
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Hello there"
    assert result.messages[1].role == "user"
    assert result.messages[1].content == "How are you?"
    assert result.messages[2].content == "I'm fine, thanks"


def test_build_history_with_mixed_entries():
    # Test with mixed entry types
    chat_history = [
        "Initial question", {
            "inputs": {
                "question": "Follow-up question"
            },
            "outputs": {
                "answer": {
                    "content": "Response to follow-up"
                }
            }
        }
    ]

    result = ChatHistoryProcessor.build_history(chat_history,
                                                "Another question")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 4
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Initial question"
    assert result.messages[1].role == "user"
    assert result.messages[1].content == "Follow-up question"
    assert result.messages[3].content == "Another question"


def test_build_history_with_missing_fields():
    # Test with dictionaries missing some fields
    chat_history = [
        {
            "inputs": {}
        },  # Missing question
        {
            "outputs": {
                "answer": {
                    "content": "Response without question"
                }
            }
        },  # Missing inputs
        {
            "inputs": {
                "question": "Question without answer"
            }
        },  # Missing outputs
        {
            "outputs": {
                "answer": {}
            }
        }  # Missing content
    ]

    result = ChatHistoryProcessor.build_history(chat_history, "Final question")

    assert isinstance(result, ChatHistory)
    # Expected to have 3 messages based on actual behavior
    assert len(result.messages) == 3
    assert result.messages[0].role == "assistant"
    assert result.messages[0].content == "Response without question"
    assert result.messages[1].role == "user"
    assert result.messages[1].content == "Question without answer"
    assert result.messages[2].role == "user"
    assert result.messages[2].content == "Final question"


def test_build_history_exception_handling(mocker):
    # Test exception handling
    mocker.patch(
        'semantic_kernel.contents.chat_history.ChatHistory.add_user_message',
        side_effect=Exception("Test exception"))

    with pytest.raises(Exception):
        ChatHistoryProcessor.build_history([{
            "inputs": {
                "question": "Question"
            }
        }], "Prompt")


def test_build_history_with_string_answer():
    # Test with answer as a string instead of a dictionary
    chat_history = [{
        "inputs": {
            "question": "What is Python?"
        },
        "outputs": {
            "answer": "Python is a programming language."
        }
    }]

    result = ChatHistoryProcessor.build_history(chat_history, "Tell me more")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 3
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "What is Python?"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "Python is a programming language."
    assert result.messages[2].content == "Tell me more"


def test_build_history_with_mixed_answer_types():
    # Test with a mixture of string and dictionary answers
    chat_history = [{
        "inputs": {
            "question": "What is Python?"
        },
        "outputs": {
            "answer": "Python is a programming language."
        }
    }, {
        "inputs": {
            "question": "Is it easy to learn?"
        },
        "outputs": {
            "answer": {
                "content": "Yes, Python is relatively easy to learn."
            }
        }
    }]

    result = ChatHistoryProcessor.build_history(chat_history, "Tell me more")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 5
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "Python is a programming language."
    assert result.messages[3].role == "assistant"
    assert result.messages[
        3].content == "Yes, Python is relatively easy to learn."


def test_build_history_with_null_answer():
    # Test with None as an answer
    chat_history = [{
        "inputs": {
            "question": "What is Python?"
        },
        "outputs": {
            "answer": None
        }
    }]

    result = ChatHistoryProcessor.build_history(chat_history, "Tell me more")

    assert isinstance(result, ChatHistory)
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "What is Python?"
    assert result.messages[1].content == "Tell me more"
