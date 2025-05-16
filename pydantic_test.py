from typing import Union
from pydantic import BaseModel
from pydantic_ai import Agent

class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str

agent: Agent[None, Union[Box, str]] = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=Union[Box, str],
    system_prompt=(
        "Extract the dimensions of a box"        
        ),
)

result = agent.run_sync('The box is 10x20x30 cm')

if isinstance(result.output, Box):
    box = result.output.model_dump()
else:
    box = {"error": result.output}

print(box)