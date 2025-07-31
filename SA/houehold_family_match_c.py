import pandas as pd
from collections import defaultdict
import itertools
import random

class HouseholdBuilder:
    def __init__(self, constraints=None, constraints_household=None):
        """
        Initialize the HouseholdBuilder with optional family and household constraints.
        :param constraints: Dictionary of constraints for families.
        :param constraints_household: Dictionary of constraints for households.
        """
        self.constraints = constraints if constraints else {}
        self.constraints_household = constraints_household if constraints_household else {}

    def set_constraints(self, constraints, constraints_household):
        """
        Set constraints for families and households.
        :param constraints: Dictionary of family constraints.
        :param constraints_household: Dictionary of household constraints.
        """
        self.constraints = constraints
        self.constraints_household = constraints_household

    def group_by_constraints(self, people, keys):
        """
        Group people by specified constraint keys.
        :param people: List of dictionaries representing people.
        :param keys: List of column names to group by.
        :return: Dictionary where keys are tuples of constraint values, and values are lists of people.
        """
        grouped = defaultdict(list)
        for person in people:
            # Create a tuple of the key values for grouping
            group_key = tuple(person[key] for key in keys)
            grouped[group_key].append(person)
        return grouped

    def calculate_family_score(self, family, person):
        """
        Calculate a compatibility score for adding a person to a family.
        Higher scores indicate better matches.
        :param family: Current family being built (list of people).
        :param person: The person to evaluate.
        :return: Compatibility score (higher is better).
        """
        score = 0

        # Example scoring logic based on constraints. Adjust as needed.
        if self.constraints.get("check_columns", True):
            columns_to_check = self.constraints.get("columns", [])
            column_weights = self.constraints.get("column_weights", {})
            for column in columns_to_check:
                family_values = set(f[column] for f in family if column in f)
                weight = column_weights.get(column, 1)  # Default weight is 1 if not specified
                if not family_values or person[column] in family_values:
                    score += weight  # Reward for matching the column value

        # Add scoring logic for family size
        max_family_size_key = self.constraints.get("max_family_size", None)
        if max_family_size_key:
            if isinstance(max_family_size_key, str):  # If max_family_size is a column name
                max_family_size = person.get(max_family_size_key, float("inf"))
                if len(family) + 1 <= max_family_size:
                    score += 5  # Add a reward for fitting within family size constraints
            elif isinstance(max_family_size_key, (int, float)):  # If max_family_size is a fixed number
                if len(family) + 1 <= max_family_size_key:
                    score += 5

        return score

    def calculate_household_score(self, families, household_constraints):
        """
        Calculate a compatibility score for a household based on families and constraints.
        :param families: List of families being evaluated.
        :param household_constraints: Dictionary of household constraints.
        :return: Compatibility score (higher is better).
        """
        total_size = sum(len(family) for family in families)
        score = 0

        # Check household size constraint
        if "household_size" in household_constraints:
            household_size_column = household_constraints["household_size"]
            size_limit = families[0][0].get(household_size_column, float("inf"))  # Assume all members share the same limit

            if total_size > size_limit:
                return -1  # Invalid household (too large)
            elif total_size < size_limit:
                # Penalize households that are smaller than the required size
                score -= (size_limit - total_size)
            else:
                # Reward households that exactly match the required size
                score += 10

        # Check compatibility for household-level columns
        if "columns" in household_constraints:
            columns_to_check = household_constraints["columns"]
            column_weights = household_constraints.get("column_weights", {})
            for column in columns_to_check:
                column_values = set(person[column] for family in families for person in family)
                # Reward if all members in the household have the same value for the column
                if len(column_values) == 1:
                    score += column_weights.get(column, 1)

        return score

    import random

    def match_people_to_households_and_families(self, people):
        """
        Match people to households and families within households using compatibility scoring.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs and family IDs.
        """
        # Keys to group by for households
        constraint_keys = ['AREAENUM', 'persons_in_household', 'families_in_household']

        # Group people by constraints
        grouped_people = self.group_by_constraints(people, constraint_keys)

        households = defaultdict(list)  # Stores households with their members
        household_id = 1  # Start household IDs from 1
        family_id_counter = 1  # Global family ID counter

        for group_key, group_members in grouped_people.items():
            areaenum, persons_in_household, families_in_household = group_key

            # Total number of people in the group
            total_people = len(group_members)

            # Calculate the number of families and households
            num_families = max(1, total_people // (persons_in_household // families_in_household))
            num_households = max(1, total_people // persons_in_household)

            # Shuffle people for random assignment
            random.shuffle(group_members)

            # Step 1: Assign people to families
            family_groups = []
            for i in range(num_families):
                family_size = min(persons_in_household // families_in_household, len(group_members))
                family = group_members[:family_size]
                group_members = group_members[family_size:]  # Remove assigned people
                for person in family:
                    person["family_id"] = family_id_counter
                family_groups.append(family)
                family_id_counter += 1

            # Step 2: Assign families to households
            for i in range(num_households):
                household_families = family_groups[:families_in_household]
                family_groups = family_groups[families_in_household:]  # Remove assigned families
                household_people = [person for family in household_families for person in family]
                for person in household_people:
                    person["household_id"] = household_id
                households[household_id].extend(household_people)
                household_id += 1

            # Step 3: Handle any remaining people or families
            # If there are leftover families, assign them to new households
            while family_groups:
                household_families = family_groups[:families_in_household]
                family_groups = family_groups[families_in_household:]
                household_people = [person for family in household_families for person in family]
                for person in household_people:
                    person["household_id"] = household_id
                households[household_id].extend(household_people)
                household_id += 1

            # If there are leftover individuals (unlikely but possible), assign them to new households
            while group_members:
                household_people = group_members[:persons_in_household]
                group_members = group_members[persons_in_household:]
                for person in household_people:
                    person["household_id"] = household_id
                    person["family_id"] = family_id_counter
                households[household_id].extend(household_people)
                household_id += 1
                family_id_counter += 1

        # Flatten the households dictionary into a single list of people
        result = []
        for members in households.values():
            result.extend(members)

        return result


    def match_people_to_households_and_families_just_family_cost(self, people):
        """
        Match people to households and families within households using compatibility scoring.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs and family IDs.
        """
        # Keys to group by for households
        constraint_keys = ['AREAENUM', 'persons_in_household', 'families_in_household']

        # Group people by constraints
        grouped_people = self.group_by_constraints(people, constraint_keys)

        households = defaultdict(list)  # Stores households with their members
        household_id = 1  # Start household IDs from 1
        family_id_counter = 1  # Global family ID counter

        for group_key, group_members in grouped_people.items():
            areaenum, persons_in_household, families_in_household = group_key

            # Total number of people in the group
            total_people = len(group_members)

            # Step 1: Assign individuals to families
            family_groups = defaultdict(list)
            for person in group_members:
                best_family_id = None
                best_score = -1

                # Match the person to the best family
                for family_id, family_members in family_groups.items():
                    score = self.calculate_family_score(family_members, person)
                    if score > best_score:
                        best_score = score
                        best_family_id = family_id

                # Assign the person to the best family or create a new family
                if best_family_id is not None and best_score > 0:
                    family_groups[best_family_id].append(person)
                    person["family_id"] = best_family_id
                else:
                    # Create a new family
                    family_groups[family_id_counter].append(person)
                    person["family_id"] = family_id_counter
                    family_id_counter += 1

            # Step 2: Assign families to households
            available_people = [person for family in family_groups.values() for person in family]
            available_families = list(family_groups.values())
            current_person_index = 0

            while current_person_index < len(available_people):
                household = []
                household_size = 0
                families_in_this_household = 0

                # Add people to the household while respecting constraints
                for i in range(current_person_index, len(available_people)):
                    person = available_people[i]
                    household.append(person)
                    household_size += 1

                    # Check if we have reached the household limits #TODO ASSING THE BEST FAMILY TO THE HOUSEHOLD OR CREATE NEW HOUSHOLD
                    if household_size >= persons_in_household or families_in_this_household >= families_in_household:
                        break

                # Assign household ID to all members in this household
                for person in household:
                    person["household_id"] = household_id
                households[household_id].extend(household)
                household_id += 1

                # Move the index forward
                current_person_index += len(household)

        # Flatten the households dictionary into a single list of people
        result = []
        for members in households.values():
            result.extend(members)

        return result

if __name__ == "__main__":
    # Load your data
    source = r'Z:\Population_Synth\census_13_06_0926_KurtisConnor\data\DATGAN_samples.csv'
    #source = 'DATGAN_samples.csv'

    data = pd.read_csv(source)

    # Sample the dataset (e.g., 10% of records)
    sampled_df = data.sample(frac=1.00).reset_index(drop=True)

    # Define family constraints
    constraints = {
        "max_family_size": 'CPRF',  # Maximum number of people per family
        "check_columns": True,
        "columns": ['CACF', 'CPRF'],
        "column_weights": {
            'CACF': 2,
            'CPRF': 4
        }
    }

    # Define household constraints
    constraints_household = {
        "household_size": "persons_in_household",  # Use the column in the dataset to enforce the constraint
        "columns": ["BEDRD", "HHCD", "VEHRD", "MRERD", 'RNTRD', 'HIED', "STRD", 'TENDLLD', 'HINASD', 'DWTD'],  # Columns to check for compatibility
        "column_weights": {
            "BEDRD": 5,
            "VEHRD": 8,
            "MRERD": 10,
            "RNTRD": 10,
            "HIED": 6,
            "STRD": 3,
            "HHCD": 5,
            "DWTD": 5
        }
    }

    # Build households and families
    builder = HouseholdBuilder(constraints, constraints_household)
    result = builder.match_people_to_households_and_families(sampled_df.to_dict(orient="records"))

    # Convert result back to DataFrame and save
    result_df = pd.DataFrame(result)
    print(result_df)
    result_df.to_csv('hhid_fid.csv', index=False)
