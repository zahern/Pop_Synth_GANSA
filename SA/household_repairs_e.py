import pandas as pd
import random

class RecordRepairer:
    def __init__(self, data):
        """
        Initialize with a dataset.
        :param data: A pandas DataFrame representing the dataset.
        """
        self.data = data

    def remove_underfilled_households(self):
        """
        Remove records belonging to underfilled households from self.data.
        """
        print("Removing underfilled households...")

        # Step 1: Calculate household sizes
        household_sizes = self.data.groupby("household_id").size().reset_index(name="actual_size")

        # Step 2: Merge the actual sizes back into the dataset
        self.data = self.data.merge(household_sizes, on="household_id", how="left")

        # Step 3: Calculate the initial number of records
        initial_count = len(self.data)

        # Step 4: Filter out underfilled households
        self.data = self.data[self.data["actual_size"] >= self.data["persons_in_household"]]

        # Step 5: Calculate the final number of records
        final_count = len(self.data)

        # Step 6: Drop the helper column "actual_size"
        self.data = self.data.drop(columns=["actual_size"])

        # Log the number of records removed
        print(f"Inital Total records of  {initial_count} belonging households.")
        removed_count = initial_count - final_count
        print(f"Removed {removed_count} records belonging to underfilled households.")
        print(f"Remaining records  of {final_count}.")


    def check_and_repair(self, row):
        """
        Check and repair a single record (row) based on constraints.
        :param row: A pandas Series representing a single record.
        :return: The repaired row or None if the record should be removed.
        """
        # Example 1: If SPIP (e.g., age) is 0 and FINASF (income) is high, repair by setting income to 0
        if row["SPIP"] == 0 and row["FINASF"] > 0:
            print(f"Repairing FINASF for record ID {row['id']} due to SPIP=0.")
            row["FINASF"] = 0

        # Example 2: If persons_in_household exceeds household_id sum, remove the record
        if row["persons_in_household"] > row["household_id_sum"]:
            print(f"Removing record ID {row['id']} due to persons_in_household exceeding household_id_sum.")
            return None

        # No other constraints violated, return the repaired row
        return row

    def repair_household_consistency_old(self):
        """
        Ensure that certain columns are consistent within the same household_id.
        Columns to standardize: BEDRD, DWTD, HHCD, VEHRD.
        """
        columns_to_standardize = ["BEDRD", "DWTD", "HHCD", "VEHRD", "RNTRD", "MRERD", "HIED", "STRD", "TENLLD", "HINASD"]
        columns_to_standardize = [col for col in columns_to_standardize if col in self.data.columns]
        # Group by household_id and apply the mode for each column
        for column in columns_to_standardize:
            # Find the mode for each household_id
            modes = self.data.groupby("abshid")[column].agg(lambda x: self.pick_mode(x))

            # Map the mode back to the original DataFrame to make all values in the household consistent
            self.data[column] = self.data["abshid"].map(modes)

    def repair_household_consistency(self):
        """
        Ensure that certain columns are consistent within the same household_id.
        If multiple modes exist in any column, create a new family ID and assign new person IDs.
        """
        print("Repairing household consistency...")

        # Columns to standardize
        columns_to_standardize = ["BEDRD", "DWTD", "HHCD", "VEHRD", "RNTRD", "MRERD", "HIED", "STRD", "TENLLD",
                                  "HINASD"]
        columns_to_standardize = [col for col in columns_to_standardize if col in self.data.columns]

        # Initialize family_id if not already present
        if "family_id" not in self.data.columns:
            self.data["family_id"] = self.data["absfid"]
        if "household_id" not in self.data.columns:
            self.data["household_id"] = self.data["abshid"]
        if "person_id" not in self.data.columns:
            self.data["person_id"] = self.data["abspid"]



        # Loop through columns to standardize
        for column in columns_to_standardize:
            # Find the mode for each household_id
            modes = self.data.groupby("household_id")[column].agg(lambda x: self.pick_mode(x))

            # Detect households with multiple modes
            households_with_multiple_modes = self.data.groupby("household_id")[column].apply(
                lambda x: len(x.mode()) > 1)

            # Assign new family IDs for households with multiple modes
            for household_id, has_multiple_modes in households_with_multiple_modes.items():
                if has_multiple_modes:
                    print(
                        f"Multiple modes detected for household {household_id} in column {column}. Creating new family IDs...")
                    household_rows = self.data[self.data["household_id"] == household_id]

                    # Create new family IDs for each row in the household
                    self.data.loc[self.data["household_id"] == household_id, "family_id"] = [
                        f"{household_id}_{i + 1}" for i in range(len(household_rows))
                    ]

            # Map the mode back to the original DataFrame to make all values in the household consistent
            self.data[column] = self.data["household_id"].map(modes)

        # Assign unique person IDs within each family
        self.data["person_id"] = (
                self.data.groupby("family_id")
                .cumcount() + 1
        )

    def pick_mode(self, series):
        """
        Pick the mode of a series. If there are multiple modes, pick one randomly.
        :param series: A pandas Series.
        :return: The mode value or a randomly chosen mode if multiple exist.
        """
        mode = series.mode()
        if len(mode) > 1:
            # Multiple modes exist, pick one randomly
            return random.choice(mode)
        return mode.iloc[0]


    def alt_repair(self):
        self.repair_household_consistency()

        # Remove underfilled households
        #self.remove_underfilled_households()

        return self.data

    def repair_dataset(self):
        """
        Repair the entire dataset by applying constraints and standardizing households.
        :return: A cleaned pandas DataFrame.
        """
        # Add a helper column for household_id sum
        self.data["household_id_sum"] = self.data.groupby("household_id")["household_id"].transform("sum")

        # Apply checks and repairs row by row
        repaired_data = self.data.apply(self.check_and_repair, axis=1)

        # Remove rows that were flagged as None
        repaired_data = repaired_data.dropna().reset_index(drop=True)

        # Drop the helper column
        repaired_data = repaired_data.drop(columns=["household_id_sum"])

        # Apply household consistency repairs
        self.data = repaired_data
        self.repair_household_consistency()

        # Remove underfilled households
        self.remove_underfilled_households()

        return self.data


# Example usage
if __name__ == "__main__":
    # Sample data (replace this with your actual dataset)
    

    # Create a DataFrame
    # whats is this, where do i grab this from.
    #hhid_fid_fixed.whY?
    df = pd.read_csv('hhid_fid_fixed.csv')

    # Instantiate the RecordRepairer
    repairer = RecordRepairer(df)

    # Repair the dataset
    cleaned_df = repairer.alt_repair()

    # Output the cleaned dataset
    print("Cleaned DataFrame:\n", cleaned_df)

    # Save the cleaned dataset to a new CSV file
    cleaned_df.to_csv("cleaned_dataset_hhid.csv", index=False)
