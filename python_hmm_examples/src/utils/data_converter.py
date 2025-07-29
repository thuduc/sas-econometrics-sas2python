"""Utility functions for converting SAS data files to CSV format."""

import pandas as pd
import pyreadstat
from pathlib import Path
from typing import Optional, Dict, Any


def convert_sas_to_csv(
    sas_path: str,
    csv_path: str,
    encoding: str = 'utf-8',
    index: bool = False
) -> Dict[str, Any]:
    """
    Convert a SAS7BDAT file to CSV format.
    
    Parameters
    ----------
    sas_path : str
        Path to the input SAS7BDAT file
    csv_path : str
        Path for the output CSV file
    encoding : str, default='utf-8'
        Encoding for the CSV file
    index : bool, default=False
        Whether to write row index to CSV
    
    Returns
    -------
    dict
        Dictionary containing metadata about the conversion
    """
    # Read SAS file
    df, meta = pyreadstat.read_sas7bdat(sas_path)
    
    # Ensure output directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=index, encoding=encoding)
    
    # Return metadata
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'column_labels': meta.column_labels,
        'file_encoding': meta.file_encoding,
        'creation_time': meta.creation_time,
        'modification_time': meta.modification_time
    }


def read_csv_with_metadata(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV file that was converted from SAS format.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        The loaded dataframe
    """
    return pd.read_csv(csv_path)


def validate_conversion(sas_path: str, csv_path: str) -> bool:
    """
    Validate that a SAS to CSV conversion was successful.
    
    Parameters
    ----------
    sas_path : str
        Path to the original SAS file
    csv_path : str
        Path to the converted CSV file
    
    Returns
    -------
    bool
        True if conversion is valid, False otherwise
    """
    try:
        # Read original SAS file
        df_sas, _ = pyreadstat.read_sas7bdat(sas_path)
        
        # Read converted CSV file
        df_csv = pd.read_csv(csv_path)
        
        # Check dimensions match
        if df_sas.shape != df_csv.shape:
            return False
        
        # Check column names match
        if list(df_sas.columns) != list(df_csv.columns):
            return False
        
        # Check data matches (allowing for floating point precision)
        for col in df_sas.columns:
            if df_sas[col].dtype in ['float64', 'float32']:
                if not pd.Series(df_sas[col]).round(6).equals(
                    pd.Series(df_csv[col]).round(6)
                ):
                    return False
            else:
                if not df_sas[col].equals(df_csv[col]):
                    return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


if __name__ == "__main__":
    # Example conversion
    sas_file = "../../data/raw/finitelearn.sas7bdat"
    csv_file = "../../data/csv/finitelearn.csv"
    
    if Path(sas_file).exists():
        metadata = convert_sas_to_csv(sas_file, csv_file)
        print(f"Converted {sas_file} to {csv_file}")
        print(f"Metadata: {metadata}")
        
        if validate_conversion(sas_file, csv_file):
            print("Conversion validated successfully!")
        else:
            print("Conversion validation failed!")