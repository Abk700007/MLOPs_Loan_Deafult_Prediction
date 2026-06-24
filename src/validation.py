import pandas as pd
import great_expectations as ge
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates the dataset using Great Expectations.
    Supports both legacy (<1.0.0) and new (>=1.0.0) versions of Great Expectations.
    Returns True if data passes basic requirements, False otherwise.
    """
    if df.empty:
        logging.error("Validation failed: DataFrame is empty.")
        return False
        
    logging.info("Starting data validation using Great Expectations...")
    
    # Check if using Great Expectations 1.0+ or legacy version
    is_legacy = hasattr(ge, "from_pandas")
    
    if is_legacy:
        logging.info("Detected legacy Great Expectations API (< 1.0.0). Using from_pandas wrapper.")
        # Wrap pandas DataFrame as a Great Expectations PandasDataset
        ge_df = ge.from_pandas(df)
        
        validation_results = []
        
        # Expect SK_ID_CURR to exist, be unique, and not be null
        validation_results.append(ge_df.expect_column_to_exist("sk_id_curr"))
        validation_results.append(ge_df.expect_column_values_to_be_unique("sk_id_curr"))
        validation_results.append(ge_df.expect_column_values_to_not_be_null("sk_id_curr"))
        
        # Expect TARGET column to exist and be either 0 or 1
        validation_results.append(ge_df.expect_column_to_exist("target"))
        validation_results.append(ge_df.expect_column_values_to_be_in_set("target", [0, 1]))
        
        # Expect financial amount columns to be positive
        validation_results.append(ge_df.expect_column_values_to_be_between("amt_income_total", min_value=0))
        validation_results.append(ge_df.expect_column_values_to_be_between("amt_credit", min_value=0))
        validation_results.append(ge_df.expect_column_values_to_be_between("amt_annuity", min_value=0))
        
        # Expect Days of Birth to be negative
        validation_results.append(ge_df.expect_column_values_to_be_between("days_birth", max_value=0))
        
        # Expect External Source score 2 and 3 to be between 0 and 1 (if not null)
        validation_results.append(ge_df.expect_column_values_to_be_between("ext_source_2", min_value=0, max_value=1))
        validation_results.append(ge_df.expect_column_values_to_be_between("ext_source_3", min_value=0, max_value=1))
        
        # Check overall validation status
        all_success = True
        for res in validation_results:
            if not res.get("success", False):
                logging.warning(f"Validation failed for check: {res.get('expectation_config', {}).get('expectation_type')}")
                logging.warning(f"Details: {res.get('result')}")
                all_success = False
                
        if all_success:
            logging.info("Data validation completed successfully. All expectations met.")
        else:
            logging.warning("Data validation completed with warnings/failures.")
            
        return all_success
    else:
        logging.info("Detected Great Expectations 1.0+ API. Using Ephemeral Context and ValidationDefinition.")
        try:
            # Ephemeral data context initialization
            context = ge.get_context(mode="ephemeral")
            datasource = context.data_sources.add_pandas(name="ds_validation")
            asset = datasource.add_dataframe_asset(name="asset_validation")
            bd = asset.add_batch_definition_whole_dataframe("bd_validation")
            
            # Create expectation suite
            suite = context.suites.add(ge.ExpectationSuite(name="suite_validation"))
            
            # Add expectations
            suite.add_expectation(ge.expectations.ExpectColumnToExist(column="sk_id_curr"))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeUnique(column="sk_id_curr"))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToNotBeNull(column="sk_id_curr"))
            suite.add_expectation(ge.expectations.ExpectColumnToExist(column="target"))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column="target", value_set=[0, 1]))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column="amt_income_total", min_value=0))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column="amt_credit", min_value=0))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column="amt_annuity", min_value=0))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column="days_birth", max_value=0))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column="ext_source_2", min_value=0, max_value=1))
            suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column="ext_source_3", min_value=0, max_value=1))
            
            # Create validation definition
            val_def = context.validation_definitions.add(
                ge.ValidationDefinition(name="val_def", data=bd, suite=suite)
            )
            
            # Run validation
            res = val_def.run(batch_parameters={"dataframe": df})
            
            if res.success:
                logging.info("Data validation completed successfully. All expectations met (GE 1.0+).")
            else:
                logging.warning("Data validation completed with warnings/failures (GE 1.0+).")
                # Log failed expectations details
                for run_res in res.run_results.values():
                    for validation_res in run_res.get("results", []):
                        if not validation_res.get("success", False):
                            logging.warning(f"Validation failed for check: {validation_res.get('expectation_config', {}).get('expectation_type')}")
                            logging.warning(f"Details: {validation_res.get('result')}")
                            
            return res.success
        except Exception as e:
            logging.error(f"Error during GE 1.0+ validation: {e}")
            return False

if __name__ == "__main__":
    test_data = pd.DataFrame({
        "sk_id_curr": [100001, 100002, 100003],
        "target": [0, 1, 0],
        "amt_income_total": [120000.0, 90000.0, 135000.0],
        "amt_credit": [450000.0, 312682.0, 254700.0],
        "amt_annuity": [20567.0, 15000.0, 18562.5],
        "days_birth": [-12000, -18000, -15000],
        "ext_source_2": [0.65, 0.45, 0.72],
        "ext_source_3": [0.49, 0.22, 0.5]
    })
    
    success = validate_data(test_data)
    print(f"Validation test success: {success}")
