
APPLICATION_PROFILES = {
    "code.exe": {
        "category": "development",
        "description": "Visual Studio Code",
        "task_hints": [
            "writing or editing code",
            "debugging",
            "navigating project files",
            "using the integrated terminal",
        ],
    },
    "devenv.exe": {
        "category": "development",
        "description": "Visual Studio",
        "task_hints": [
            "software development",
            "debugging",
            "building projects",
        ],
    },
    "unity.exe": {
        "category": "game_development",
        "description": "Unity Editor",
        "task_hints": [
            "editing scenes",
            "developing game systems",
            "inspecting assets",
            "running the game",
        ],
    },
    "unrealeditor.exe": {
        "category": "game_development",
        "description": "Unreal Editor",
        "task_hints": [
            "editing levels",
            "working with Blueprints",
            "creating game systems",
            "testing gameplay",
        ],
    },
    "chrome.exe": {
        "category": "web_browsing",
        "description": "Google Chrome",
        "task_hints": [
            "browsing websites",
            "researching information",
            "using web applications",
        ],
    },
    "msedge.exe": {
        "category": "web_browsing",
        "description": "Microsoft Edge",
        "task_hints": [
            "browsing websites",
            "researching information",
            "using web applications",
        ],
    },
}


def get_application_profile(process_name: str) -> dict:
    if not process_name:
        return {
            "category": "unknown",
            "description": "Unknown application",
            "task_hints": [],
        }

    return APPLICATION_PROFILES.get(
        process_name.lower(),
        {
            "category": "other",
            "description": process_name,
            "task_hints": [],
        },
    )
