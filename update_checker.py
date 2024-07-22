"""
To Do:
- Define the necessary elements of a 'complete' update
- Determine which elements are 
"""
# update_checker.py
from langchain.schema import BaseTool, ToolOutput

class UpdateChecker(BaseTool):
    def __init__(self):
        super().__init__(
            name="UpdateChecker", 
            description="Checks the user's response to determine if it should be considered a complete update.")

    def invoke(self, response: str) -> ToolOutput:
        required_elements = [
            "user's name",
            "project the user is working on",
            "user's recent accomplishments or achievements",
            "issues or blockers",
            "significant risks",
            "notable personal updates"
        ]
        
        missing_elements = [element for element in required_elements if element not in response.lower()]
        
        if missing_elements:
            return ToolOutput(content=f"The response is missing the following elements: {', '.join(missing_elements)}")
        else:
            return ToolOutput(content="The response contains all the required elements. Great job!.")
