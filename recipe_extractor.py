from typing import List, Union
from pydantic import BaseModel
from pydantic_ai import Agent
import json
from collections import defaultdict
import requests
from markdownify import markdownify as md

def fetch_and_convert_to_markdown(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    html_content = response.text

    markdown = md(html_content)
    return markdown

def run_agent(input_text):
    class Ingredient(BaseModel):
        name: str
        quantity: float
        unit: str

    class Recipe(BaseModel):
        name: str
        servings: int
        ingredients: List[Ingredient]

    agent: Agent[None, Union[Recipe, str]] = Agent(
        'google-gla:gemini-1.5-flash',
        output_type=Union[Recipe, str],
        system_prompt=(
            "Extract a recipe from the text."
            "Return the recipe name, number of servings, and a list of ingredients."
            "Each ingredient should include name, quantity, and unit."
        ),
    )

    result = agent.run_sync(input_text)

    if isinstance(result.output, Recipe):
        recipe = result.output.model_dump()
    else:
        recipe = {"error": result.output}

    return recipe

def combine_ingredients(ingredients):
    combined = defaultdict(float)

    for ing in ingredients:
        key = (ing["name"].strip().lower(), ing["unit"].strip().lower())
        combined[key] += ing["quantity"]

    return [
        {"name": name, "quantity": round(quantity, 2), "unit": unit}
        for (name, unit), quantity in combined.items()
    ]

def main():
    url = input("Enter a URL: ")
    markdown_output = fetch_and_convert_to_markdown(url)
    recipe = run_agent(markdown_output)
    recipe["ingredients"] = combine_ingredients(recipe["ingredients"])
    pretty_json_final = json.dumps(recipe, indent=4)
    print(pretty_json_final)

if __name__ == "__main__":
    main()