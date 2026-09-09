Dataset

This project uses the Indian Real Estate – 99acres.com dataset from Kaggle, which contains property listings scraped from 99acres.com across several major Indian cities. For this project, the dataset was filtered down to listings from Gurgaon, focusing specifically on sector-based residential properties (~10,000+ listings) to keep the price-prediction problem scoped to a single, well-defined market.

The raw data required significant cleaning before use — prices and areas were stored as inconsistent text (e.g. "1.2 Cr", "45 Lakh", "1200 sqft", "133 sq.yards"), and locality names needed to be parsed to extract sector numbers.
