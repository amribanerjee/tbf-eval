import json
import os

def create_mock_fixtures():
    os.makedirs('tbf/data/system_gpt4', exist_ok=True)
    os.makedirs('tbf/data/system_claude3', exist_ok=True)

    mock_data = [
        {
            "instance_id": f"django__test_issue_{i}",
            "model": "gpt-4-archive-run" if i % 2 == 0 else "claude-3-archive-run",
            "success": True if i % 3 == 0 else False,
            "history": [
                {"action": "cd /workspace/django", "observation": "Directory changed"},
                {"action": f"find_files.py --search 'auth_user_{i}'", "observation": "Found 1 file"},
                {"action": "view_file.py --lines 1-50", "observation": "Displaying file content..."},
                {"action": "edit_file.py --line 12 --text 'return True'", "observation": "File modified successfully"},
                {"action": "run_tests.py --suite auth_tests", "observation": "Tests passed" if i % 3 == 0 else "Tests failed"}
            ]
        } for i in range(1, 13)
    ]

    with open('tbf/data/system_gpt4/trajectories.json', 'w') as f:
        json.dump(mock_data[:6], f, indent=2)

    with open('tbf/data/system_claude3/trajectories.json', 'w') as f:
        json.dump(mock_data[6:], f, indent=2)

if __name__ == "__main__":
    create_mock_fixtures()
