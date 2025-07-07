import os
import pandas as pd

def create_attribute(table_name, column_name, values):
    return {
        'table_name': table_name,
        'column_name': column_name,
        'values': sorted(set(str(v) for v in values if pd.notna(v))),
        'full_name': f"{table_name}.{column_name}"
    }

def get_min_value(attribute):
    return attribute['values'][0] if attribute['values'] else float('inf')

def load_csv_files(directory_path):
    attributes = []

    csv_files = [f for f in os.listdir(directory_path)]

    print(f"Found {len(csv_files)} \n CSV files: {csv_files}")

    for filename in csv_files:
        file_path = os.path.join(directory_path, filename)
        table_name = os.path.splitext(filename)[0]

        df = pd.read_csv(file_path)
        print(f"Processing {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

        for column in df.columns:
            non_null_values = df[column].dropna().tolist()
            if non_null_values:
                attr = create_attribute(table_name, column, non_null_values)
                if attr['values']:
                    attributes.append(attr)
                print(f"Added attribute: {attr['full_name']} ({len(attr['values'])} unique values)")

    return attributes

def spider_algorithm(attributes):
    if not attributes:
        return set()

    print(f"{len(attributes)=}")

    attributes.sort(key=get_min_value)

    cursors = {attr['full_name']: 0 for attr in attributes}
    attr_map = {attr['full_name']: attr for attr in attributes}
    value_sets = {attr['full_name']: set(attr['values']) for attr in attributes}

    total_comparisons = 0
    all_inds = set()

    while any(cursors[name] < len(attr_map[name]['values']) for name in cursors):
        valid_attrs = [name for name in cursors if cursors[name] < len(attr_map[name]['values'])]
        if not valid_attrs:
            break

        min_name = min(valid_attrs, key=lambda name: attr_map[name]['values'][cursors[name]])
        current_value = attr_map[min_name]['values'][cursors[min_name]]

        for other_name in attr_map:
            if min_name != other_name:
                total_comparisons += 1
                if current_value in value_sets[other_name]:
                    all_inds.add((min_name, other_name))

        cursors[min_name] += 1

        if total_comparisons % 10000 == 0:
            print(f"  Processed {total_comparisons=}")

    print(f"{total_comparisons=}")
    return all_inds

def filter_valid_inclusion_dependencies(inclusion_candidates, attributes):
    print(f"\n{len(inclusion_candidates)=}")

    attr_map = {attr['full_name']: attr for attr in attributes}
    value_sets = {attr['full_name']: set(attr['values']) for attr in attributes}

    valid_inds = set()

    for dep_name, ref_name in inclusion_candidates:
        dep_values = set(attr_map[dep_name]['values'])
        ref_values = value_sets[ref_name]
        if dep_values.issubset(ref_values):
            valid_inds.add((dep_name, ref_name))

    print(f"{len(valid_inds)=}")
    return valid_inds

def save_results_to_file(inclusion_dependencies, output_file):
    with open(output_file, 'w') as f:
        sorted_inds = sorted(inclusion_dependencies)
        for dep_name, ref_name in sorted_inds:
            f.write(f"{dep_name}={ref_name}\n")

    print(f"Total inclusion dependencies found: {len(inclusion_dependencies)}")


def main():
    csv_directory = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind-db"
    output_file = "/home/haseeb/Desktop/EKAI/ERD_automation/codes/inclusionDependencyWithSpider/spider_results/northwind-db.txt"

    attributes = load_csv_files(csv_directory)

    print(f"\nLoaded {len(attributes)} attributes from CSV files")

    inclusion_candidates = spider_algorithm(attributes)
    valid_inclusion_dependencies = filter_valid_inclusion_dependencies(inclusion_candidates, attributes)

    save_results_to_file(valid_inclusion_dependencies, output_file)

if __name__ == "__main__":
    main()
