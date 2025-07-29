tables_to_analyze = "REGION, CATEGORIES, EMPLOYEE_TERRITORIES, SUPPLIERS, US_STATES, ORDERS, ORDER_DETAILS, TERRITORIES, PRODUCTS, EMPLOYEES, SHIPPERS, CUSTOMERS"
tables_context = '''
CUSTOMERS - Customer information with contact details and location
ORDERS - Order transactions with shipping and date information
ORDER_DETAILS - Individual line items for each order
PRODUCTS - Product catalog with pricing and inventory
CATEGORIES - Product categorization system
SUPPLIERS - Vendor/supplier information
EMPLOYEES - Staff information including hierarchy
EMPLOYEE_TERRITORIES - Assignment of employees to territories
TERRITORIES - Sales territory definitions
REGION - Regional divisions
SHIPPERS - Shipping companies
US_STATES - US state reference data'''
description = "The Northwind database represents a complete e-commerce/trading company system with interconnected tables for managing customers, orders, products, employees, and geographical territories. The database follows a normalized structure with clear relationships between entities, supporting order management, inventory tracking, employee territory assignments, and supplier relationships."
format_instructions = """
Provide output in the following JSON format:
{
    "table_name": "Clear description of the table's purpose",
}
"""

DOCUMENTATION_PROMPT_TEMPLATE = """You are an Expert Data Engineer. Your task is to write clear descriptions for database tables and their columns to document an undocumented database.
<Context>
The descriptions you generate will be used to:
1. Help identify primary and foreign keys
2. Establish table relationships
3. Document the database structure
</Context>

Additional Information:
- Give the user provided information the highest priority in all the information you have.

<Tables To Process>
{tables_to_analyze}
</Tables To Process>

<Tables Context>
{tables_context}
</Tables Context>

<Description>
Use the information below to generate an accurate description. 
{description}

**Note:** If there's are comments already available in it, use that.
**Note:** Create a meaningful description for the table, do not output the DDL as it is.
</Description>

<Response Format>
{format_instructions}
- Tone should be certain for descriptions.
- Keep description to the point.
</Response Format>

**Note:** Ensure the response is in proper JSON formatting
"""

# Now format the template with all variables
formatted_prompt = DOCUMENTATION_PROMPT_TEMPLATE.format(
    tables_to_analyze=tables_to_analyze,
    tables_context=tables_context,
    description=description,
    format_instructions=format_instructions
)

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from sentence_transformers import SentenceTransformer
import hdbscan
from collections import defaultdict
import logging

def main():
    logging.basicConfig(
                    filename="/home/haseeb/Desktop/EKAI/ERD_automation/codes/my_automation/my_automation/logs/compartmentalization.log",
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.INFO
    )
    load_dotenv("/home/haseeb/Desktop/EKAI/ERD_automation/codes/my_automation/.env")
    client = OpenAI()

    response = client.responses.create(
        model="o4-mini",
        input=formatted_prompt
    )

    table_description = json.loads(response.output_text)
    texts = [f"{key} : {value}" for key, value in table_description.items()]

    table_names = list(table_description.keys())
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusterer.fit(embeddings)
    labels = clusterer.labels_

    table_clusters = {table_names[i]: int(labels[i]) for i in range(len(table_names))}


    # Print results

    cluster_groups = defaultdict(list)
    for table, label in table_clusters.items():
        cluster_groups[label].append(table)
    
    logging.info(f"{"-"*50} Evaluating HDBSCAN {"-"*50}")

    for label, tables in cluster_groups.items():
        logging.info(f"\nCluster {label}:")
        for t in tables:
            logging.info(f" - {t}")

if __name__ == "__main__":
    main()