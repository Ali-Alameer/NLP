from api import OpenFoodFactsAPI

# List of categories to fetch products from. These are the category slugs used in the Open Food Facts API. You can find more categories at https://world.openfoodfacts.org/categories.json
CATEGORIES = [
    "meats",
    "seafood",
    "dairies",
    "beverages",
    "cereals-and-potatoes",
    "fruits",
    "vegetables",
    "alcoholic-beverages",
    "dietary-supplements",
    "waters",
]

api = OpenFoodFactsAPI(
    page_size=50,
    max_products_per_category=500,
    category_slugs=CATEGORIES,
    workers=2,
    username=None,
    password=None
)

# Fetch products for all categories
api.fetch_all_products()

# Fetch products for a single category (e.g., "seafood")
# api.fetch_single_category("seafood")
