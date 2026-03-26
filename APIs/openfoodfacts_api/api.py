from typing import Optional
from utils import (
    TokenBucket,
    retry_on_exception,
    time_it,
    upsert_jsonl_dataset,
)
import requests
from concurrent.futures import ThreadPoolExecutor


class OpenFoodFactsAPI:
    def __init__(
            self,
            page_size: int = 20,
            max_products_per_category: int = 10,
            category_slugs: list = None,
            workers: int = 5,
            additional_fields: list = None,
            jsonl_file_path: str = "openfoodfacts_products.jsonl",
            username: Optional[str] = None,
            password: Optional[str] = None,
    ):
        self.base_url = "https://world.openfoodfacts.org"
        # The Search API has a rate limit of 10 requests per minute, so we set a rate of 0.16 req/sec (1 req every 6.25 seconds) and a small burst capacity of 2 tokens.
        self.token_bucket = TokenBucket(rate_per_sec=0.16, capacity=2)
        self.page_size = page_size
        self.start_page = 1
        self.current_page = None
        self.max_products_per_category = max_products_per_category
        self.current_category_index = None
        self.jsonl_file_path = jsonl_file_path
        self.category_slugs = category_slugs
        self.workers = workers
        self.max_pages = self.workers * 2  # Fetch 2 pages per worker
        self.additional_fields = additional_fields
        self.session = requests.Session()
        if username and password:
            self.login(username, password)

    def login(self, username: str, password: str) -> None:
        """Authenticate with the Open Food Facts API. Required for write operations.

        Args:
            username (str): Open Food Facts username.
            password (str): Open Food Facts password.

        Raises:
            Exception: If login fails.
        """
        response = self.session.post(
            f"{self.base_url}/cgi/session.pl",
            data={"user_id": username, "password": password},
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"},
            timeout=30,
        )
        if response.status_code != 200 or "wrong_password" in response.text or "incorrect" in response.text.lower():
            raise Exception(
                f"Login failed: {response.status_code} - {response.text[:200]}")
        print("\033[92mLogged in to Open Food Facts successfully.\033[0m")

    @time_it
    @retry_on_exception(max_retries=3)
    def _get_category_slug(self) -> list:
        """Method to get the list of available category slugs from the Open Food Facts API."""

        print("\033[96mFetching category slugs from API...\033[0m")
        response = self.session.get(
            f"{self.base_url}/categories.json", timeout=300)
        if response.status_code != 200:
            raise Exception(
                f"\033[93mAPI error: {response.status_code} - {response.text}\033[0m"
            )
        data = response.json()
        category_slugs = [category["id"] for category in data.get("tags", [])]
        return category_slugs

    def get_category_slug(self) -> None:
        """Method to fetch category slugs from the API and store them in the instance variable."""
        self.category_slugs = self._get_category_slug()
        print(
            f"\033[96mFetched {len(self.category_slugs)} category slugs from API.\033[0m"
        )

    def get_filter_params(self, category: str) -> dict:
        feilds = [
            "product_name",
            "product_name_en",
            "generic_name_en",
            "brands",
            "labels_tags",
            "categories_tags_en",
            "packaging_text",
            "food_groups_tags",
            "nutriments",
            "allergens_tags",
            "additives_tags",
            "lang",
            "languages_tags",
            "ingredients_text",
        ]
        if self.additional_fields:
            feilds.extend(self.additional_fields)

        params = dict(
            categories_tags_en=category,
            fields=",".join(feilds),
        )

        return params

    @time_it
    @retry_on_exception(max_retries=3)
    def search_api_request(self, params: dict) -> dict:
        """Method to send GET request to Search API to fetch products data

        Args:
            params (dict): Query parameters of the request

        Raises:
            Exception: If the API response status code is not 200, an exception is raised with the error message.

        Returns:
            dict: The JSON response from the API parsed into a Python dictionary.
        """
        self.token_bucket.acquire()
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        }
        response = self.session.get(
            f"{self.base_url}/api/v2/search", params=params, headers=headers, timeout=300
        )
        if response.status_code != 200:
            raise Exception(
                f"\033[93mAPI error: {response.status_code} - {response.text}\033[0m"
            )
        return response.json()

    def search(self, category: str, page: int = 1) -> dict:
        """Method to fetch products by page

        Args:
            category (str): Product category to search for.
            page (int, optional): Page number to fetch. Defaults to 1.

        Returns:
            dict: Search results.
        """
        params = self.get_filter_params(category)
        params["page"] = page
        params["page_size"] = self.page_size
        print(
            f"\033[96mFetching page {page} for category '{category}'...\033[0m")
        return self.search_api_request(params)

    @time_it
    def paginated_search(self, category: str) -> None:
        """Method to fetch products for a category with pagination and save results to parquet file.

        Args:
            category (str): Category slug to search for (e.g., 'meats').
            max_pages (int, optional): Maximum number of pages to fetch. Defaults to 5.
        """

        # Run paginated search in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.workers) as executor:

            self.current_page = self.start_page
            products_saved = 0

            while products_saved < self.max_products_per_category:
                futures = []

                for step in range(self.current_page, self.current_page + self.max_pages):
                    futures.append(executor.submit(
                        self.search, category, step))

                self.current_page += self.max_pages

                # Get completed results from futures and save to parquet file
                for future in futures:
                    try:
                        results: dict = future.result()

                        # If there are no products in the results, it means we've reached the end of available pages for this category, so we can stop fetching further pages.
                        if len(results.get("products", [])) == 0:
                            print(
                                f"\033[96mNo more results available after page {self.current_page - 1}.\033[0m"
                            )
                            return

                        products = []

                        # Filter and parse products from the API response
                        if products := results.get("products", []):
                            # If adding the parsed products would exceed the max products limit for this category, trim the list to fit the remaining allowed products.
                            if products_saved + len(products) > self.max_products_per_category:
                                products = products[:self.max_products_per_category - products_saved]

                            # Upsert parsed products to parquet file
                            upsert_jsonl_dataset(
                                products, self.jsonl_file_path)

                            # Update the count of products saved for this category
                            products_saved += len(products)

                        else:
                            print(
                                f"\033[96mNo valid products to save for category '{category}' on this page after filtering.\033[0m"
                            )
                            continue

                        if products_saved >= self.max_products_per_category:
                            print(
                                f"\033[92mReached max products for category '{category}' after page {self.current_page - 1}.\033[0m"
                            )
                            return

                    except Exception as e:
                        print(f"\033[93mError fetching page: {e}\033[0m")

    def fetch_all_products(self) -> None:
        """Method to fetch products for all categories defined in the category_slugs list."""

        # Fetch category slugs from API if not already provided in the constructor
        if self.category_slugs is None:
            self.get_category_slug()

        for category in self.category_slugs:
            self.fetch_single_category(category)
        print("\033[92mFinished fetching products for all categories.\033[0m")

    def fetch_single_category(self, category: str) -> None:
        """Method to fetch products for a single category."""

        assert self.category_slugs is not None, "Category slugs list is not initialized. Call get_category_slug() first or provide a category_slugs list in the constructor."
        if category not in set(self.category_slugs):
            print(
                f"\033[93mCategory '{category}' is not in the predefined category_slugs list.\033[0m")
            return
        print(
            f"\033[96mFetching products for category '{category}'...\033[0m")
        self.current_category_index = self.category_slugs.index(category)
        self.paginated_search(category)


if __name__ == "__main__":
    # Example usage: Fetch products for all categories defined in the CATEGORIES list

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
        page_size=20, max_products_per_category=10, category_slugs=CATEGORIES)
    api.fetch_all_products()
