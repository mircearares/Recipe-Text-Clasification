import requests
import re
import json
from bs4 import BeautifulSoup


def format_data(data):
    punct = ['.', '?', '!', ',', ';', ':']

    recipe_data = ""

    for item in data:
        words = item.split()

        for word in words:
            if word[-1] in punct and len(word) > 1:
                if word[-2] != " ":
                    end = word[-1]
                    word = word[:-1] + " " + end

            recipe_data += (word + " ")

    return recipe_data


def export_recipes(cuisine_pages, cuisines):
    json_to_export = list()
    counter = 7458

    for index in range(0, len(cuisine_pages)):
        for no_page in range(1, 30):
            link = cuisine_pages[index] + "?page=" + str(no_page)

            webpage = requests.get(link)
            soup = BeautifulSoup(webpage.content, features="html.parser")

            links = soup.find_all("article", {"class": "fixed-recipe-card"})

            try:
                for item in links:
                    recipe = dict()

                    ingredients_list = list()
                    directions_list = list()

                    print(item.a["href"])

                    new_webpage = requests.get(item.a["href"])
                    new_soup = BeautifulSoup(new_webpage.content, features="html.parser")

                    title = re.split(" Recipe - ", new_soup.title.text)[0]

                    directions = new_soup.find_all("span", {"class": "recipe-directions__list--item"})
                    ingredients = new_soup.find_all("span", {"class": "recipe-ingred_txt added"})

                    for direction in directions:
                        direction = direction.text.replace("\n", "")

                        if direction != "":
                            directions_list.append(direction)

                    for ingredient in ingredients:
                        ingredients_list.append(ingredient.next)

                    recipe["id"] = counter
                    recipe["title"] = title
                    recipe["cuisine"] = cuisines[index]
                    recipe["ingredients"] = ingredients_list
                    recipe["directions"] = format_data(directions_list)

                    json_to_export.append(recipe)
                    counter += 1
            except Exception:
                with open("text_southern.json", "w+") as outfile:
                    json.dump(json_to_export, outfile)

    with open("text_9.json", "w+") as outfile:
        json.dump(json_to_export, outfile)


if __name__ == '__main__':
    cuisine_page_list = ["https://www.allrecipes.com/recipes/15876/us-recipes/southern/"
                         # "https://www.allrecipes.com/recipes/272/us-recipes/cajun-and-creole/"
                         # "https://www.allrecipes.com/recipes/696/world-cuisine/asian/filipino/",
                         # "https://www.allrecipes.com/recipes/702/world-cuisine/asian/thai/",
                         # "https://www.allrecipes.com/recipes/700/world-cuisine/asian/korean/",
                         # "https://www.allrecipes.com/recipes/233/world-cuisine/asian/indian/",
                         # "https://www.allrecipes.com/recipes/699/world-cuisine/asian/japanese/",
                         # "https://www.allrecipes.com/recipes/695/world-cuisine/asian/chinese/",
                         # "https://www.allrecipes.com/recipes/230/world-cuisine/latin-american/caribbean/",
                         # "https://www.allrecipes.com/recipes/728/world-cuisine/latin-american/mexican/",
                         # "https://www.allrecipes.com/recipes/730/world-cuisine/latin-american/south-american/",
                         # "https://www.allrecipes.com/recipes/236/us-recipes/",
                         # "https://www.allrecipes.com/recipes/17582/world-cuisine/african/north-african/",
                         # "https://www.allrecipes.com/recipes/15035/world-cuisine/african/south-african/",
                         # "https://www.allrecipes.com/recipes/723/world-cuisine/european/italian/",
                         # "https://www.allrecipes.com/recipes/731/world-cuisine/european/greek/",
                         # "https://www.allrecipes.com/recipes/721/world-cuisine/european/french/",
                         # "https://www.allrecipes.com/recipes/726/world-cuisine/european/spanish/",
                         # "https://www.allrecipes.com/recipes/722/world-cuisine/european/german/",
                         # "https://www.allrecipes.com/recipes/704/world-cuisine/european/uk-and-ireland/",
                         # "https://www.allrecipes.com/recipes/712/world-cuisine/european/eastern-european",
                         # "https://www.allrecipes.com/recipes/725/world-cuisine/european/scandinavian/",
                         # "https://www.allrecipes.com/recipes/235/world-cuisine/middle-eastern/"
                         ]

    cuisine_list = ["southern-us"
                    # "cajun-creole"
                    # "filipino"
                    # "thai", "korean", "indian", "japanese", "chinese", "caribbean",
                    # "mexican", "south-american", "north-american", "north-african", "south-african",
                    # "italian", "greek", "french", "spanish", "german",
                    # "british", "eastern-european", "scandinavian", "middle-eastern"
                    ]

    export_recipes(cuisine_page_list, cuisine_list)
