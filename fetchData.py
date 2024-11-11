import os
import pandas as pd
import h5py
import re

file_extension ={'CWAM' : 'h5',
                 'FUSER' :'csv',
                 'METAR' : 'txt', 
                 'TAF' : 'txt'}

username = os.getlogin()
home_dir = 'home'
project_name = 'finalProject'
data_dir_file =  'data'
defult_base_dir = os.path.join('/', home_dir, username, project_name, data_dir_file)

def get_defult_base_dir():
    return defult_base_dir

# Parse METAR report 
def parse_metar_line(date_time, line):
    """Parse a single METAR report line into its components, including optional AUTO."""
    metar_pattern = re.compile(
        r'^(?P<station>[A-Z]{4})\s+'
        r'(?P<datetime>\d{2}\d{4}Z)\s+'
        r'(?P<auto>AUTO\s+)?'
        r'(?P<wind>\d{3}\d{2}(G\d{2})?(KT|MPS|KMH))?\s*'
        r'(?P<visibility>\d{4}(SM|NDV)?|CAVOK|////)?\s*'
        r'(?P<clouds>(///\d{3}///|FEW|SCT|BKN|OVC|NSC|VV|///)\d{0,3}[A-Z]{0,3}\s*)*'
        r'(?P<temperature>M?\d{2})/(?P<dewpoint>M?\d{2}|//)\s+'
        r'Q(?P<pressure>\d{4})\s*'
        r'(?P<remarks>.+)?$'
    )
    
    match = metar_pattern.match(line)
    if match:
        data = match.groupdict()
        
        # Process the clouds group to handle multiple cloud layers
        clouds = data.get('clouds')
        if clouds:
            data['clouds'] = clouds.strip().split()
        else:
            data['clouds'] = None
        
        data['date_time'] = date_time
        # Handle optional fields
        data['auto'] = data['auto'].strip() if data['auto'] else None
        data['remarks'] = data['remarks'].strip() if data['remarks'] else None
        
        return data
    
    return None

# Parse TAF report 
def parse_taf_block(block):
    """Parse a full TAF block into its components."""
    # Regex pattern to match main TAF components across multiple lines
    taf_pattern = re.compile(
        r'(?P<station>TAF \w{4})\s+'                     # Station identifier (TAF + ICAO code)
        r'(?P<datetime>\d{6}Z)\s+'                       # Date and time of issuance
        r'(?P<validity>\d{4}/\d{4})\s+'                  # Validity period
        r'(?P<wind>(\d{5}(KT|MPS|KMH)|VRB\d{2}(KT|MPS|KMH)?)?)\s*'  # Wind information
        r'(?P<visibility>CAVOK|\d{4})?\s*'               # Visibility, including CAVOK
        r'(?P<clouds>(SCT|BKN|FEW|NSC|OVC|NSC|VV)\d{3}[A-Z]{0,3})?\s*' # Cloud information
        r'(?P<temp_dewpoint>(TX|TN)\d{2}/\d{4}Z)?\s*'    # Max/min temperature forecast
        
        # Additional forecasts as separate groups
        r'(?P<probability>(PROB\d{2}\s+\d{4}/\d{4}\s+.*)*)'     # Probability forecasts (e.g., PROB30)
        r'(?P<temporary>(TEMPO\s+\d{4}/\d{4}\s+.*)*)'        # Temporary forecasts
        r'(?P<becoming>(BECMG\s+\d{4}/\d{4}\s+.*)*)'        # Becoming forecasts
        r'(?P<from_forecasts>(FM\d{6}\s+.*)*)'                       # From forecasts
    )
    
    match = taf_pattern.search(block)
    if match:
        return match.groupdict()
    return None

# CWAM
def get_dataset(f):
    data_entries = []
    for forecast_time in f['Deviation Probability']:
        for flight_level in f[f'Deviation Probability/{forecast_time}']:
            for contour in f[f'Deviation Probability/{forecast_time}/{flight_level}']:
                for threshold in f[f'Deviation Probability/{forecast_time}/{flight_level}/{contour}']:
                    for polygon in f[f'Deviation Probability/{forecast_time}/{flight_level}/{contour}/{threshold}']:
                        # Construct the full path for the dataset
                        dataset_name = f'Deviation Probability/{forecast_time}/{flight_level}/{contour}/{threshold}/{polygon}'
                        
                        dataset = f[dataset_name][:]
                        latitudes, longitudes = dataset[0], dataset[1]
                        
                        fcst = forecast_time
                        flvl = flight_level
                        trsh = threshold
                        poly = polygon

                        data_entries.append({
                            "Forecast Time (FCST)" : fcst,
                            "Flight Level (FLVL)" : flvl,
                            "Threshold (TRSH)" : trsh,
                            "Polygon Number (POLY)" : poly,
                            "Latitudes" : latitudes,
                            "Longitudes" : longitudes
                        })
    return pd.DataFrame(data_entries)

def load_data(data_type, dataset_purpose, path_level=None, month=None, day=None, file_name=None, base_dir=None):
    """
    Loads a data file from a specified path or defult path.
    
    Parameters:
    - data_type (str): Data type, e.g., "CWAM", "FUSER", "METAR", "TAF".
    - dataset_purpose (str): "train" or "test", indicating training or testing data.
    - path_level (str): CWAM or METAR part_X, e.g., "part_1".
    - month (str): Month, e.g., "09".
    - day (str): Day, e.g., "29".
    - file_name (str): Filename (without extension or type-specific suffix), e.g., "2022_09_29_20_00_GMT.Forecast".
    - base_dir (str): Root directory of the data, e.g., "/home/finalProject/data".
    
    Returns:
    - DataFrame, HDF5 file object, or string content based on file type.
    """
    base_dir = defult_base_dir if not base_dir else base_dir

    # Specify file extensions
    file_ext = file_extension[data_type] 
    
    # Build the file path based on data type
    if data_type == "CWAM":
        # CWAM Path: data/CWAM/test/part_X/MM/DD
        if not (path_level and month and day):
            raise ValueError("CWAM data requires path_level, month, and day")
        path = os.path.join(base_dir, data_type, dataset_purpose, path_level, month, day)
        # CWAM files use a specific naming convention
        file_path = os.path.join(path, f"{file_name}.h5.CWAM.h5")
        
    elif data_type == "FUSER":
        # FUSER Path: data/FUSER/train/KXXX/
        if not path_level:
            raise ValueError("FUSER data requires fuser_type")
        path = os.path.join(base_dir, data_type, dataset_purpose, path_level)
        file_path = os.path.join(path, f"{file_name}.csv")
    
    elif data_type in ["METAR", "TAF"]:
        # METAR and TAF Paths: data/METAR/train/part_X/
        if dataset_purpose == "train" and not path_level:
            raise ValueError(f"{data_type} data requires path_level")
        if dataset_purpose == "train":
            path = os.path.join(base_dir, data_type, dataset_purpose, path_level)
        else:
            path = os.path.join(base_dir, data_type, dataset_purpose)
        file_path = os.path.join(path, f"{file_name}.txt")
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load file based on file type
    if data_type == "FUSER":
        def extract_file_type(file_name):
            pattern = r'^(?P<airport>\w+)' \
                      r'_(?P<date_range>\d{4}-\d{2}-\d{2}(_\d{4}-\d{2}-\d{2})?)' \
                      r'\.(?P<file_type>\w+)_data_set$'
              
            match = re.match(pattern, file_name)
            
            if match:
                file_type = match.group('file_type')
                return file_type
            else:
                return None
        
        df = pd.read_csv(file_path)
        file_type = extract_file_type(file_name)
        df['file_type'] = file_type
        return df
    elif data_type in ["METAR", "TAF"]:
        if data_type == "METAR":
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines if line.strip()]
                data_entries = []
                date_time = None  # To keep track of the current date and time

                for line in lines:
                    # Check if the line is a date line
                    if re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}', line):
                        date_time = line  # Store the date and time
                    elif date_time:  # If we have a date_time, process the METAR line
                        parsed_data = parse_metar_line(date_time, line)
                        if parsed_data:
                            data_entries.append(parsed_data)
                        date_time = None 
            return pd.DataFrame(data_entries)
        else:
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                lines = file.readlines()
    
            data_entries = []
            buffer = ""  # To accumulate lines for a single TAF block

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # If the line starts with a date, treat it as a new TAF report block
                date_match = re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}', line)
                if date_match:
                    if buffer:  # Parse the previous buffer before moving to the next date block
                        parsed_data = parse_taf_block(buffer)
                        if parsed_data:
                            parsed_data["Date and Time"] = date_time  # Associate with the last stored date
                            data_entries.append(parsed_data)
                        buffer = ""  # Clear buffer for the new TAF block
                    date_time = date_match.group()  # Update date_time to the new block's date

                # Continue adding lines to the buffer for the current TAF block
                if line.startswith("TAF") or buffer:
                    buffer += line + " "

                # Move to the next line
                i += 1

            # Parse the last block in the buffer after the loop ends
            if buffer:
                parsed_data = parse_taf_block(buffer)
                if parsed_data:
                    parsed_data["Date and Time"] = date_time
                    data_entries.append(parsed_data)

            # Convert the list of dictionaries to a DataFrame
            return pd.DataFrame(data_entries)
    elif data_type == "CWAM":
        with h5py.File(file_path, 'r') as file:  # Load HDF5 data
            
            return get_dataset(file)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def get_fuser_file_name(airport, data_range,file_type):
    """
    Return a FUSER file name.
    
    Parameters:
    - airport (str): airport name.
    - data_range (str): a data range, eg., 2022-09-01
    - file_type (str): the file type of dataset

    Returns:
    - FUSER file name.
    """
    return f"{airport}_{data_range}.{file_type}_data_set"

def get_cwam_file_name(airport, data_range,file_type):
    """
    Return a FUSER file name.
    
    Parameters:
    - airport (str): airport name.
    - data_range (str): a data range, eg., 2022-09-01
    - file_type (str): the file type of dataset

    Returns:
    - FUSER file name.
    """
    return f"{airport}_{data_range}.{file_type}_data_set"

def check_input_file_exists(data_type, dataset_purpose, path_level=None, month=None, day=None, file_name=None, base_dir=None):
    """
    Return a file path exists or not.
    
    Parameters:
    - data_type (str): Data type, e.g., "CWAM", "FUSER", "METAR", "TAF".
    - dataset_purpose (str): "train" or "test", indicating training or testing data.
    - path_level (str): CWAM or METAR part_X, e.g., "part_1".
    - month (str): Month, e.g., "09".
    - day (str): Day, e.g., "29".
    - file_name (str): Filename (without extension or type-specific suffix), e.g., "2022_09_29_20_00_GMT.Forecast".
    - base_dir (str): Root directory of the data, e.g., "/home/finalProject/data".
    
    Returns:
    - True: a vaild file path; otherwise, False
    """

    file_ext = file_extension[data_type]
    base_dir = defult_base_dir if not base_dir else base_dir


    path = os.path.join(base_dir, data_type, dataset_purpose, path_level)
  
    file_path = os.path.join(path, f"{file_name}.{file_ext}")
    return os.path.exists(file_path)

def check_output_file_exists(data_type, dataset_purpose, month=None, day=None, file_name=None, base_dir=None):
    """
    Return a file path exists or not.
    
    Parameters:
    - data_type (str): Data type, e.g., "CWAM", "FUSER", "METAR", "TAF".
    - dataset_purpose (str): "train" or "test", indicating training or testing data.
    - path_level (str): CWAM or METAR part_X, e.g., "part_1".
    - month (str): Month, e.g., "09".
    - day (str): Day, e.g., "29".
    - file_name (str): Filename (without extension or type-specific suffix), e.g., "2022_09_29_20_00_GMT.Forecast".
    - base_dir (str): Root directory of the data, e.g., "/home/finalProject/data".
    
    Returns:
    - True: a vaild file path; otherwise, False
    """

    file_ext = file_extension[data_type]
    base_dir = defult_base_dir if not base_dir else base_dir


    path = os.path.join(base_dir, data_type, dataset_purpose)
  
    file_path = os.path.join(path, f"{file_name}.{file_ext}")
    
    return os.path.exists(file_path)