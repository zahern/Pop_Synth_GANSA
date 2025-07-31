import pandas as pd
from collections import defaultdict, deque
import time
class HouseholdBuilder:
    def __init__(self, constraints=None):
        """
        Initialize the HouseholdBuilder with optional constraints.
        :param constraints: Dictionary of constraints to apply when forming households.
        """
        self.constraints = constraints if constraints else {}

    def set_constraints(self, constraints):
        """
        Set constraints for building households.
        :param constraints: Dictionary of constraints.
        """
        self.constraints = constraints

    def validate_constraints(self, household, person):
        """
        Validate constraints for adding a person to a household.
        :param household: Current household being built.
        :param person: The person record to validate.
        :return: True if constraints are satisfied, False otherwise.
        """
        # Constraint: Maximum of 2 partners
        if self.constraints.get("max_partners", False):
            partner_count = sum(1 for p in household if p["relationship"] == "partner")
            if person["relationship"] == "partner" and partner_count >= self.constraints["max_partners"]:
                return False


        # Constraint: Maximum number of family members (dynamic based on a column)
        max_family_size_key = self.constraints.get("max_family_size", None)
        if max_family_size_key:
            # If max_family_size is a column name, use its value
            if isinstance(max_family_size_key, str):  # Check if it's a column name
                max_family_size = person.get(max_family_size_key, float("inf"))  # Default to no limit
                if len(household) + 1 > max_family_size:
                    return False
            # If max_family_size is a fixed number, validate against it
            elif isinstance(max_family_size_key, (int, float)):
                if len(household) + 1 > max_family_size_key:
                    return False

        # Constraint: Number of vehicles in a household must be consistent
        if self.constraints.get("check_vehicles", False):
            vehicles = set(p["vehicles"] for p in household)
            if len(vehicles) > 0 and person["vehicles"] not in vehicles:
                return False
        '''
        if self.constraints.get("check_columns", True):
            # Get the list of columns to check (default to ["vehicles"])
            columns_to_check = self.constraints.get("columns", ["vehicles"])

            for column in columns_to_check:
                column_values = set(p[column] for p in household)
                if len(column_values) > 0 and person[column] not in column_values:
                    return False
        '''
        # Add other constraints here as needed
        return True

    def assign_household_ids(self, people):
        """
        Assign household IDs to a list of people based on constraints.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs.
        """
        households = defaultdict(list)  # Dictionary to store households
        household_id = 1  # Start household IDs from 1

        for person in people:
            assigned = False

            # Try to assign the person to an existing household
            for h_id, members in households.items():
                if self.validate_constraints(members, person):
                    households[h_id].append(person)
                    person["household_id"] = h_id
                    assigned = True
                    break

            # If not assigned, create a new household
            if not assigned:
                households[household_id].append(person)
                person["household_id"] = household_id
                household_id += 1

        return people




    def calculate_compatibility_score(self, household, person):
        """
        Calculate a compatibility score for adding a person to a household.
        Higher scores indicate better matches.
        :param household: Current household being built.
        :param person: The person to evaluate.
        :return: Compatibility score (higher is better).
        """
        score = 0

        # Example scoring logic based on constraints:
        if self.constraints.get("max_partners", False):
            partner_count = sum(1 for p in household if p["relationship"] == "partner")
            if person["relationship"] == "partner" and partner_count < self.constraints["max_partners"]:
                score += 10  # Assign a high score for valid partner matches

        if self.constraints.get("mffax_family_size", False):
            if len(household) + 1 <= self.constraints["max_family_size"]:
                score += 5  # Reward for staying under family size limit

        if self.constraints.get("check_vehicles", False):
            vehicles = set(p["vehicles"] for p in household)
            if len(vehicles) == 0 or person["vehicles"] in vehicles:
                score += 3  # Reward for matching vehicle count

        # Generalized column checking
        if self.constraints.get("check_columns", True):
            columns_to_check = self.constraints.get("columns",
                                                    ["vehicles"])  # Default to "vehicles" if none specified
            column_weights = self.constraints.get("column_weights", {})  # Optional weights for columns
            for column in columns_to_check:
                column_values = set(p[column] for p in household)  # Collect unique values for the column
                weight = column_weights.get(column, 3)  # Default weight is 3 if not specified
                if len(column_values) == 0 or person[column] in column_values:
                    score += weight  # Reward for matching the column value


        # Add other custom scoring criteria here
        return score

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

    def build_households_fast(self, people):
        """
        Match people to households using exact matching on key constraints.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs.
        """
        # Keys to group by (the 4 key constraints)
        constraint_keys = ['families_in_household', 'persons_in_family','persons_in_household', 'AREAENUM']

        # Group people by constraint keys
        grouped_people = self.group_by_constraints(people, constraint_keys)

        households = []
        household_id = 1
        global_id = 1
        # Process each group
        for group_key, group_members in grouped_people.items():
            # Assign household IDs for each person in the group
            for person in group_members:
                person["household_id"] = household_id
            households.extend(group_members)
            household_id += 1

        return households

    ''' OLD CODE, test
    def match_people_to_households(self, people):
        """
        Match people to households using a scoring-based algorithm.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs.
        """
        households = defaultdict(list)  # Dictionary to store households
        household_id = 1  # Start household IDs from 1
        processed_count = 0
        # Keep track of unmatched people
        unmatched_people = list(people)
        print('Number of unmatched people is ', len(unmatched_people))
        start_time = time.time()  # Record the start time

        while unmatched_people:
            processed_count += 1  #
            if len(unmatched_people) % 1000 == 0:
                elapsed_time = time.time() - start_time  # Time elapsed so far
                avg_time_per_person = elapsed_time / processed_count  # Average time per person
                remaining_people = len(unmatched_people)
                estimated_time_remaining = avg_time_per_person * remaining_people  # Estimated remaining time

                # Format the estimated time in hours, minutes, and seconds
                hours, rem = divmod(estimated_time_remaining, 3600)
                minutes, seconds = divmod(rem, 60)

                print(
                    f'Processed {processed_count} people so far. '
                    f'Remaining unmatched: {remaining_people}. '
                    f'Est. time remaining: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
                )

            person = unmatched_people.pop(0)  # Take the first unmatched person
            best_match_id = None
            best_score = -1

            # Evaluate compatibility with existing households
            for h_id, members in households.items():
                score = self.calculate_compatibility_score(members, person)
                if score > best_score:
                    best_score = score
                    best_match_id = h_id

            # If a suitable household is found, assign the person
            if best_match_id is not None and best_score > 0:
                households[best_match_id].append(person)
                person["household_id"] = best_match_id
            else:
                # Otherwise, create a new household
                households[household_id].append(person)
                person["household_id"] = household_id
                household_id += 1


        # Flatten the households dictionary into a list
        result = []
        for members in households.values():
            result.extend(members)

        return result
    
    '''

    def initialize_households(self, people, constraint_keys):
        """
        Precompute households based on grouped people and constraints, ensuring each household has at least one person.
        :param people: List of dictionaries representing people.
        :param constraint_keys: Keys used to group people.
        :return: A dictionary of precomputed households, the grouped people, and the unmatched people.
        """
        # Step 1: Group people by key constraints
        grouped_people = self.group_by_constraints(people, constraint_keys)

        households = defaultdict(list)  # Dictionary to store households
        household_id = 1  # Start household IDs from 1

        unmatched_people = []  # List to store people who are not matched during initialization

        # Step 2: Precompute households based on group sizes
        for group_key, group_members in grouped_people.items():
            # Calculate the number of households needed for this group
            total_people = len(group_members)
            required_household_size = group_members[0][
                'persons_in_household']  # Assume all members in the group have the same size constraint
            num_households = (total_people + required_household_size - 1) // required_household_size  # Round up

            # Create households and assign the first person to each
            for _ in range(num_households):
                if group_members:  # Ensure there are members left in the group
                    # Assign the first person in the group to the household
                    first_person = group_members.pop(0)
                    households[household_id] = {"group_key": group_key, "members": [first_person]}
                    first_person["household_id"] = household_id  # Track the household ID for the person
                else:
                    # Create an empty household if no members are left
                    households[household_id] = {"group_key": group_key, "members": []}

                household_id += 1

            # If there are remaining members in the group, add them to unmatched_people
            unmatched_people.extend(group_members)

        return households, grouped_people, unmatched_people

    def flatten_households_and_save(self, households, it = 0):
        """
        Flatten the members from all households into a single list.
        :param households: Dictionary of households with members.
        :return: A flattened list of all members in all households.
        """
        result = []
        for household in households.values():
            # Access the "members" key in each household
            result.extend(household["members"])

        save_string = 'hhid{}.csv'.format(it)
        result = pd.DataFrame(result)
        result.to_csv(save_string)

        


    def match_people_to_households(self, people):
        """
        Match people to households using a hybrid approach: group by constraints first,
        then apply a scoring-based algorithm within each group.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs.
        """
        # Keys to group by (the 4 key constraints)
        # Add unique IDs to all people
        for i, person in enumerate(people):
            person["id"] = i  # Add a unique ID for each person


        constraint_keys = ['families_in_household', 'persons_in_family','persons_in_household', 'AREAENUM']

        # Step 1: Group people by key constraints
        #grouped_people = self.group_by_constraints(people, constraint_keys)
        ## Step 1: Precompute households and group people
        households, grouped_people, unmatched_people = self.initialize_households(people, constraint_keys)
        #households = defaultdict(list)  # Dictionary to store households
        #maybe we should take the list of households up front. it should be a dunction of grouped people
        #and number of gorups
        household_id = 1  # Start household IDs from 1
        processed_count = 0
        start_time = time.time()  # Record the start time
        global_counter = 1
        # Step 2: Process each group
        group_counter = 1
        for group_key, group_members in grouped_people.items():
            group_counter = global_counter
            unmatched_people = deque(group_members)  # Use deque for efficient popping
            #household_scores = {}
            # Initialize scores for each household in this group
            #for im, household in households.items():

               # if household["group_key"] == group_key:
                    # Create a globally unique ID using group_key and global_counter
                   # unique_id = f"{group_key}_{global_counter}"
                    #global_counter += 1
                    #household_scores[global_counter] = 0  # Initialize score


            household_scores = {h_id: 0 for h_id, household in households.items() if
                               household["group_key"] == group_key}


            best_match_id = None
            best_score = -1 #best _core should be a score for each_household
            while unmatched_people:
                processed_count += 1
                if processed_count % 10000 == 0:  # Log progress every 1000 people
                    if processed_count % 10000 == 0: #save every 10 000
                        self.flatten_households_and_save(households, processed_count)


                    elapsed_time = time.time() - start_time

                    avg_time_per_person = elapsed_time / processed_count
                    remaining_people = len(unmatched_people)
                    estimated_time_remaining = avg_time_per_person * remaining_people
                    hours, rem = divmod(estimated_time_remaining, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print(
                        f'Processed {processed_count} people so far. '
                        f'Remaining unmatched: {remaining_people}. '
                        f'Est. time remaining: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
                    )

                # Pop a person from the unmatched list, this immedicatly takes it out, we should probably do it later
                person = unmatched_people.popleft()

                hhcd = person['persons_in_family']  # Get household size requirement
                # Place the first person into a household if no existing households match



                # Step 3: Evaluate compatibility within the group
                # Filter households where hhid > group_counter
                #filtered_households = {hh_id: household for hh_id, household in households.items() if
                                      # int(hh_id) > group_counter}

                for hh_id, household in households.items():
                    household_scores[hh_id] = 0

                    if len(household) == 0:
                        continue
                    if household['group_key'] != group_key:
                            continue
                    #members = household["members"]  # Access 'member
                    members = household['members']  # Access 'members' directly
                    if not members or len(members) >= hhcd:
                        household_scores[hh_id] = 0


                    # calculate compatibility score
                    score = self.calculate_compatibility_score(household["members"], person)

                    # Update the score for this household
                    household_scores[hh_id] = score

                # Step 4: Assign person to the best matching household
                best_match_id = max(household_scores, key=household_scores.get)
                best_score = household_scores[best_match_id]

                # Step 5: Assign person to the best matching household
                if best_match_id is not None and best_score > 0:
                    # Assign the person to the best household
                    person["household_id"] = best_match_id
                    households[best_match_id]["members"].append(person)

                '''
                else:
                    # Log an error if no valid household is found (shouldn't happen if all households are created upfront)
                    print(f"Error: No valid household found for person {person['id']} in group {group_key}.")
                '''
        # Step 5: Flatten the households dictionary into a list
        result = []
        for household in households.values():
            result.extend(household['members'])

        return result

    def build_households(self, people):
        """
        Match people to households using constraints and compatibility scoring.
        :param people: List of dictionaries representing people.
        :return: List of people with assigned household IDs.
        """
        households = defaultdict(list)
        household_id = 1
        unmatched_people = deque(people)  # Use deque for efficient popping
        processed_count = 0

        print(f"Number of unmatched people: {len(unmatched_people)}")
        start_time = time.time()

        while unmatched_people:
            processed_count += 1
            person = unmatched_people.popleft()  # Efficient pop from the left
            best_match_id = None
            best_score = -1

            # Evaluate compatibility with existing households
            for h_id, members in households.items():
                if self.validate_constraints(members, person):
                    score = self.calculate_compatibility_score(members, person)
                    if score > best_score:
                        best_score = score
                        best_match_id = h_id

            # Assign to the best matching household or create a new one
            if best_match_id is not None:
                households[best_match_id].append(person)
                person["household_id"] = best_match_id
            else:
                households[household_id].append(person)
                person["household_id"] = household_id
                household_id += 1

            # Logging progress every 1000 people
            if processed_count % 1000 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_person = elapsed_time / processed_count
                remaining_people = len(unmatched_people)
                estimated_time_remaining = avg_time_per_person * remaining_people
                hours, rem = divmod(estimated_time_remaining, 3600)
                minutes, seconds = divmod(rem, 60)
                print(
                    f"Processed {processed_count} people so far. "
                    f"Remaining unmatched: {remaining_people}. "
                    f"Est. time remaining: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                )

        # Flatten households dictionary into a list
        result = []
        for members in households.values():
            result.extend(members)

        return result

# Example usage
if __name__ == "__main__":
    # Example data: a list of people
    data = [
        {"id": 1, "age": 35, "gender": "male", "relationship": "partner", "family_size": 4, "vehicles": 2},
        {"id": 2, "age": 34, "gender": "female", "relationship": "partner", "family_size": 4, "vehicles": 2},
        {"id": 3, "age": 10, "gender": "male", "relationship": "child", "family_size": 4, "vehicles": 2},
        {"id": 4, "age": 8, "gender": "female", "relationship": "child", "family_size": 4, "vehicles": 2},
        {"id": 5, "age": 40, "gender": "male", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 6, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 7, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 8, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 9, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 10, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 11, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 4},
        {"id": 12, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 13, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 1},
        {"id": 14, "age": 39, "gender": "female", "relationship": "partner", "family_size": 2, "vehicles": 2}
    ]
    source = 'Z:/brisbane_gan/data/DATGAN.csv'
    #maybe
    source = 'DATGAN_samples.csv'
    data = pd.read_csv(source)
    # Convert to DataFrame for easier manipulation (optional)
    df = pd.DataFrame(data)
    sampled_df = df.sample(frac = 0.99)
    # Reset the index if needed
    df = sampled_df.reset_index(drop=True)
    # i want to slit my data so its only 10% of the records how do i do this/

    # Define constraints
    constraints = {
        #"max_partners": 2,
        "max_family_size": 'CPRF',  # Maximum number of people per household
        "check_columns": True,
        "columns": ["BEDRD", "HHCD", "VEHRD", "MRERD", 'RNTRD', 'HIED', 'CACF', 'CPRF', 'STRD', 'TENDLLD', 'HINASD'],
        "column_weights": {
            "BEDRD": 5,  # Give more importance to "vehicles"
            "VEHRD": 2,  # Give less importance to "age_group"
            "BEDRD": 5,
            "MRERD":10,
            'RNTRD':10,
            'HIED':3,
            'CACF': 2,
            'CPRF': 4,
            'STRD': 4,
            'TENDLLD': 4,
            'HINASD': 5

        }
        # Add additional constraints here if needed
    }

    # Build households
    builder = HouseholdBuilder(constraints)
    result = builder.match_people_to_households(df.to_dict(orient="records"))

    # Convert result back to DataFrame
    result_df = pd.DataFrame(result)

    print(result_df)
    result_df.to_csv('hhid.csv')
