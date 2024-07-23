"""
To Do:
- Define the necessary elements of a 'complete' update
- Determine which elements are missing
"""
# update_checker.py
from langchain.tools import BaseTool

class UpdateChecker(BaseTool):
    name = "UpdateChecker"
    description = "Checks the user's response to determine if it contains relevant update information."

    def _run(self, response: str) -> str:
        required_elements = [
            "name",
            "project",
            "accomplishments",
            "blockers",
            "risks",
            "personal updates"
        ]
        
        found_elements = [element for element in required_elements if element.lower() in response.lower()]
        missing_elements = [element for element in required_elements if element not in found_elements]
        
        if missing_elements:
            return f"The update is missing information about: {', '.join(missing_elements)}."
        else:
            return "The update contains information about all required elements."

    async def _arun(self, response: str) -> str:
        return self._run(response)