#!/bin/bash

curl --location -N --no-buffer 'http://127.0.0.1:5000/score' \
--header 'Content-Type: application/json' \
--header 'Accept: text/event-stream' \
--data '{
    "question": "Compute 1 + 1.",
    "chat_history": []
}' | while read -r line; do
    # Skip empty lines
    if [[ -n "$line" ]]; then
        echo "→ Raw data: $line"
        # If the line begins with "data: ", extract the JSON part
        if [[ "$line" =~ ^data:\ (.*) ]]; then
            json_data="${BASH_REMATCH[1]}"
            # Print formatted JSON if jq is available
            if command -v jq &> /dev/null; then
                echo "→ Parsed JSON:"
                echo "$json_data" | jq .
            fi
        fi
    fi
done