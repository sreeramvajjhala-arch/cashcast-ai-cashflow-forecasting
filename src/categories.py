"""Shared category vocabulary and lightweight merchant heuristics."""

CATEGORY_KEYWORDS = {
    "Food": ("zomato", "swiggy", "restaurant", "cafe", "grocer", "blinkit"),
    "Rent": ("rent", "landlord", "housing"),
    "Transport": ("uber", "ola", "metro", "fuel", "petrol", "taxi"),
    "Shopping": ("amazon", "flipkart", "myntra", "mall", "store"),
    "Utilities": ("electricity", "water", "broadband", "mobile bill", "gas"),
    "Entertainment": ("netflix", "spotify", "movie", "bookmyshow", "game"),
    "Health": ("pharmacy", "clinic", "doctor", "hospital", "fitness"),
    "Education": ("course", "udemy", "books", "exam", "tuition"),
    "Travel": ("flight", "hotel", "rail", "airbnb", "trip"),
    "Income": ("salary", "credit interest", "refund", "freelance", "bonus"),
}

EXPENSE_CATEGORIES = tuple(category for category in CATEGORY_KEYWORDS if category != "Income")

