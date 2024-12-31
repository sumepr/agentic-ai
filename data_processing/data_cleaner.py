import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
import re
from typing import Union, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.datetime_columns = [
            'REVIEW_FIRSTPUBLISHTIME',
            'REVIEW_LASTMODIFICATIONTIME',
            'REVIEW_SUBMISSIONTIME'
        ]
        self.numeric_columns = ['REVIEW_RATING', 'REVIEW_RATINGRANGE']
        self.text_columns = ['REVIEW_REVIEWTEXT', 'REVIEW_TITLE']
        
    def clean_data(self) -> pd.DataFrame:
        """Main function to clean the dataset"""
        self._clean_datetime_columns()
        self._clean_numeric_columns()
        self._clean_text_columns()
        self._clean_categorical_columns()
        self._handle_missing_values()
        return self.df
    
    def _clean_datetime_columns(self):
        """Clean and convert datetime columns"""
        for col in self.datetime_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
    
    def _clean_numeric_columns(self):
        """Clean numeric columns"""
        for col in self.numeric_columns:
            if col in self.df.columns:
                # Convert to numeric, coerce errors to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # Handle out of range values
                if col == 'REVIEW_RATING':
                    self.df.loc[~self.df[col].between(1, 5), col] = np.nan
    
    def _clean_text_columns(self):
        """Clean text columns"""
        for col in self.text_columns:
            if col in self.df.columns:
                # Remove special characters and extra whitespace
                self.df[col] = self.df[col].astype(str).apply(self._clean_text)
                
                # Remove empty strings and convert to NaN
                self.df.loc[self.df[col].str.strip() == '', col] = np.nan
    
    def _clean_categorical_columns(self):
        """Clean categorical columns"""
        categorical_columns = [
            'PRODUCT_SKU_CODE', 'PRODUCT_ACTIVE_INDICATOR',
            'PRODUCT_PRICING_GROUP_NAME', 'PRODUCT_GROUP_NAME',
            'PRODUCT_ROLL_UP_NAME', 'PRODUCT_SAMPLE_INDICATOR'
        ]
        
        for col in categorical_columns:
            if col in self.df.columns:
                # Convert to string type
                self.df[col] = self.df[col].astype(str)
                
                # Clean strings
                self.df[col] = self.df[col].str.strip().str.upper()
                
                # Handle empty strings
                self.df.loc[self.df[col] == '', col] = np.nan
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Numeric columns: impute with median
        num_imputer = SimpleImputer(strategy='median')
        self.df[self.numeric_columns] = num_imputer.fit_transform(
            self.df[self.numeric_columns]
        )
        
        # Categorical columns: impute with mode
        cat_columns = self.df.select_dtypes(include=['object']).columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df[cat_columns] = cat_imputer.fit_transform(self.df[cat_columns])
        
        # Datetime columns: forward fill then backward fill
        self.df[self.datetime_columns] = self.df[self.datetime_columns].fillna(
            method='ffill'
        ).fillna(method='bfill')
    
    @staticmethod
    def _clean_text(text: Union[str, float]) -> str:
        """Clean individual text entries"""
        if pd.isna(text):
            return ''
            
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def validate_data(self) -> List[str]:
        """Validate the cleaned dataset"""
        issues = []
        
        # Check for remaining missing values
        missing_vals = self.df.isnull().sum()
        if missing_vals.any():
            issues.append(f"Missing values found in columns: {missing_vals[missing_vals > 0].index.tolist()}")
        
        # Validate rating range
        if 'REVIEW_RATING' in self.df.columns:
            invalid_ratings = self.df['REVIEW_RATING'].isin(range(1, 6))
            if not invalid_ratings.all():
                issues.append("Invalid ratings found outside range 1-5")
        
        # Validate datetime columns
        for col in self.datetime_columns:
            if col in self.df.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    issues.append(f"{col} is not in datetime format")
        
        return issues

# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('your_data.csv')
    
    # Initialize and run cleaner
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean_data()
    
    # Validate results
    issues = cleaner.validate_data()
    if issues:
        print("\nData validation issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nData cleaning completed successfully with no validation issues.")
