import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
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

# Define cloud cover dictionary for sky cover conversion
cloud_cover_dict = {
    "SKC": 0,  # Sky Clear
    "CLR": 0,  # Clear
    "NSC": 0,  # No Significant Clouds
    "NCD": 0,  # No Cloud Detected
    "FEW": 1,  # Few (1/8 to 2/8 sky cover)
    "SCT": 3,  # Scattered (3/8 to 4/8 sky cover)
    "BKN": 5,  # Broken (5/8 to 7/8 sky cover)
    "OVC": 8,  # Overcast (8/8 sky cover)
    "VV": 9    # Vertical Visibility (obscured sky, treated as full overcast)
}

def parse_cloud_layers(cloud_layers):
    """Convert cloud layer codes to structured data for modeling."""
    parsed_layers = []
    for layer in cloud_layers:
        # Match cloud layer code, altitude, and CB flag if present
        match = re.match(r"([A-Z]{3})(\d{3})?(CB)?", layer)
        if match:
            cloud_code, altitude, cumulonimbus = match.groups()
            sky_cover = cloud_cover_dict.get(cloud_code, None)  # Get sky cover value
            altitude_ft = int(altitude) * 100 if altitude else None  # Convert altitude to feet if present
            cb_flag = 1 if cumulonimbus else 0  # Cumulonimbus flag (1 for CB, 0 otherwise)

            # Append the structured cloud data
            parsed_layers.append({
                "sky_cover": sky_cover,           # Numerical sky cover level
                "altitude_ft": altitude_ft,       # Altitude in feet
                "cumulonimbus": cb_flag           # Cumulonimbus presence (1 or 0)
            })
    return parsed_layers

def get_defult_base_dir():
    return defult_base_dir

# Enhanced METAR pattern to capture complex remarks and additional fields
metar_pattern = re.compile(
    r'^(?P<station>[A-Z0-9]{4})\s+'                        # Station code
    r'(?P<datetime>\d{2}\d{4}Z)\s+'                     # Date and time
    r'(?P<cor>COR\s+)?'                                 # Optional correction indicator
    r'(?P<auto>AUTO\s+)?'                               # Optional AUTO indicator
    r'(?P<wind>(VRB|\d{3}|/////)\d{2}(G\d{2})?(KT|MPS|KMH)?\s*(\d{3}V\d{3})?)?\s*'  # Wind information
    r'(?P<visibility>////|CAVOK|\d{4}(SM|NDV)?|[0-9]+SM)?\s*'  # Visibility
    r'(?P<weather>[\+\-]?[A-Z]{2,6}\s*)?'               # Optional weather phenomena
    r'(?P<clouds>((FEW|SCT|BKN|OVC|NSC|VV|NCD|CLR|SKC|///)\d{0,3}(CB)?\s*)*)'  # Cloud layers
    r'(?:(?P<temperature>M?\d{2}|//)/(?P<dewpoint>M?\d{2}|//)\s+)?'    # Optional temperature and dewpoint
    r'(?:(?P<pressure_indicator>[QA])(?P<pressure_value>\d{4}|////)(=)?\s*)?'  # Optional pressure with optional '='
    r'(?P<has_rmk>RMK\s+)?'                            # Optional RMK section
)

# Parse METAR report
def parse_metar_line(date_time, line):
    match = metar_pattern.match(line)
    if match:
        data = match.groupdict()

        # Convert `date_time` to UTC format
        try:
            utc_date = datetime.strptime(date_time, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
            data['date_time'] = utc_date.strftime("%Y-%m-%d %H:%M:%S")
            data['date_time'] = pd.to_datetime(data['date_time'], utc=True)
        except ValueError as e:
            data['date_time'] = None

        # AUTO handling
        data['auto'] = True if data.get('clouds') == 'AUTO' else False

        # Process cloud layers into structured data
        clouds = data.get('clouds')
        data['cloud_layers'] = parse_cloud_layers(clouds.strip().split()) if clouds else []

        # Temperature conversion
        if data.get('temperature'):
            data['temperature'] = float(data['temperature'].replace("M", "-")) if data['temperature'] != "//" else None
        else:
            data['temperature'] = None

        if data.get('dewpoint'):
            data['dewpoint'] = float(data['dewpoint'].replace("M", "-")) if data['dewpoint'] and data['dewpoint'] != "//" else None
        else:
            data['dewpoint'] = None

        # Visibility conversion (to meters)
        visibility = data.get('visibility')
        if visibility == "CAVOK":
            data['visibility_meters'] = 10000  # Convention for CAVOK
        elif visibility and "SM" in visibility:
            visibility_miles = float(visibility.replace("SM", ""))
            data['visibility_meters'] = int(visibility_miles * 1609.34)
        elif visibility and visibility.isdigit():
            data['visibility_meters'] = int(visibility)
        else:
            data['visibility_meters'] = None

        # Wind speed conversion (to m/s)
        wind = data.get('wind')
        if wind and "/////" not in wind:  # Handle missing wind speed
            wind_speed_match = re.search(r'\d{2}', wind)
            wind_speed = int(wind_speed_match.group()) if wind_speed_match else 0
            if "KT" in wind:
                data['wind_speed_mps'] = round(wind_speed * 0.514444, 2)
            elif "KMH" in wind:
                data['wind_speed_mps'] = round(wind_speed / 3.6, 2)
            elif "MPS" in wind:
                data['wind_speed_mps'] = wind_speed
            else:
                data['wind_speed_mps'] = None
        else:
            data['wind_speed_mps'] = None

        # Pressure handling with unit conversion
        pressure_indicator = data.get('pressure_indicator')
        pressure_value = data.get('pressure_value')
        if pressure_value and pressure_value != "////":
            if pressure_indicator == "A":
                # Convert inHg (A) to hPa (Q) by multiplying by 33.8639
                data['pressure'] = round(int(pressure_value) * 33.8639 / 100, 2)
            elif pressure_indicator == "Q":
                data['pressure'] = int(pressure_value)
            else:
                data['pressure'] = None
        else:
            data['pressure'] = None

        fields_to_drop = ['wind', 'clouds', 'visibility', 'pressure_indicator', 'pressure_value', 'has_rmk']
        for field in fields_to_drop:
            if field in data:
                del data[field]
        return data
    return None

taf_pattern = re.compile(
    r'^(?:PART\s+\d+(?:\s+OF(?:\s+\d+)?)?\s+)?'                           # Optional PART X OF Y
    r'(?:TAF\s+)*'                                               # Allow multiple TAF prefixes
    r'(?P<amended>(AMD)\s+)?'                                 # Optional AMD flag for amended reports
    r'(?P<corection>(COR)\s+)?'                                 # Optional COR flag for corrected reports
    r'(?P<station>\w{4})(?:\s+\w{4})?\s+'                                      # ICAO station code
    r'(?P<issue_datetime>\d{6}Z\s+)?'                             # Optional issue date and time
    r'(?P<validity>(?P<valid_start_day>\d{2})(?P<valid_start_hour>\d{2})/'
    r'(?P<valid_end_day>\d{2})(?P<valid_end_hour>\d{2}))\s+'      # Validity period
    r'(?P<wind>(VRB|\d{3})\d{2,3}(G\d{2,3})?[/|?]?(KT|MPS|KMH)?)?\s*' # Wind information
    r'(?P<visibility>\d{4}|CAVOK|9999|////|[0-9]+SM)?\s*'         # Visibility

    # Weather with optional intensity prefix (+ or -) and non-greedy match
    r'(?P<weather>(\+|-)?[A-Z]{2,6})?\s+'                         # Weather phenomena with optional intensity, followed by whitespace
    r'(?P<clouds>((FEW|SCT|BKN|OVC|NSC|VV|NCD|CLR|SKC|///)\d{3}(CB)?\s*)*)' # Cloud layers
    r'(?P<qnh>QNH\d{4}(INS|HPA)?)?\s*'                            # QNH (altimeter setting) in inches or hPa
    r'(?P<variable_wind>WND\s+\d{3}V\d{3})?\s*'                   # Variable wind direction

    # Max/Min Temperature capture with optional minus sign for temperatures
    r'(?P<max_temp>TX(?P<max_temp_value>M?\d{2})/'                # Max temperature with optional minus sign
    r'(?P<max_temp_day>\d{2})(?P<max_temp_hour>\d{2})Z\s*)?'
    r'(?P<min_temp>TN(?P<min_temp_value>M?\d{2})/'                # Min temperature with optional minus sign
    r'(?P<min_temp_day>\d{2})(?P<min_temp_hour>\d{2})Z\s*)?'
    r'(?P<probability>PROB(?P<prob_value>\d{2}))?\s*'             # Probability

    # Capture everything else in additional_sections
    r'(?P<additional_sections>.*?)$'                               # Capture all remaining parts as additional sections
)

from datetime import datetime, timedelta
import pandas as pd

def calculate_temperature_dates(base_date, valid_day, valid_hour):
    # Parse base_date as the initial date in UTC
    date_time = pd.to_datetime(base_date).tz_convert('UTC')

    # Adjust for overflow into the next month if the day exceeds the month's range
    try:
        # Try to set the valid date based on the provided valid day and hour
        temp_date = date_time.replace(day=valid_day, hour=valid_hour, minute=0, second=0)
    except ValueError:
        # Overflow into the next month: move to the 1st of the next month at 00:00
        next_month = (date_time.month % 12) + 1
        year_increment = 1 if next_month == 1 else 0
        temp_date = date_time.replace(
            year=date_time.year + year_increment,
            month=next_month,
            day=1,
            hour=0,
            minute=0,
            second=0
        )

    return temp_date

def process_validity(data):
    # Ensure 'date_time' is a datetime object in UTC for calculations
    base_date = pd.to_datetime(data['date_time'], utc=True)

    # Construct 'valid_start_date'
    start_day = int(data['valid_start_day'])
    start_hour = int(data['valid_start_hour'])
    try:
        valid_start_date = base_date.replace(day=start_day, hour=start_hour)
    except ValueError:
        # Handle invalid day (e.g., 31 in a 30-day month)
        valid_start_date = base_date + relativedelta(months=1, day=start_day, hour=start_hour)

    # Adjust month and year if necessary for valid_start_date
    while valid_start_date < base_date:
        valid_start_date += relativedelta(months=1)

    # Construct 'valid_end_date'
    end_day = int(data['valid_end_day'])
    end_hour = int(data['valid_end_hour'])

    # If end_hour is 24, roll over to the next day at 0:00
    end_hour %= 24
    end_day += end_hour //24

    # Initialize valid_end_date with appropriate month rollover and validate day
    try:
        valid_end_date = base_date.replace(day=1, hour=end_hour, minute=0, second=0) + relativedelta(days=end_day - 1)
    except ValueError:
        # Adjust day if it exceeds the number of days in the new month
        valid_end_date = base_date + relativedelta(months=1, day=1, hour=end_hour) + timedelta(days=end_day - 2)

    # Ensure valid_end_date is after valid_start_date, adjusting month if necessary
    while valid_end_date < valid_start_date:
        valid_end_date += relativedelta(months=1)

    # Add results back to the data dictionary
    data['valid_start_date'] = valid_start_date
    data['valid_end_date'] = valid_end_date
    return data

def process_wind(data):
    """Processes the wind field, extracts direction and speed, and converts units to knots if necessary."""
    # Extract the wind field
    wind_str = data.get('wind')
    if wind_str is None:
        # Set default values for missing wind data
        data['has_wind'] = 0
        data['wind_is_variable'] = 0
        data['wind_direction'] = -2
        data['wind_speed_kt'] = None
        data['wind_gust_kt'] = None
        return data

    # Use regex to match and extract components of the wind string
    match = re.match(r'^(VRB|\d{3})?([/?]?\d{2,3})(G[/?]?\d{2,3})?(KT|MPS|KMH)?$', wind_str)
    if not match:
        # Set defaults for unrecognized wind patterns
        data['has_wind'] = 0
        data['wind_is_variable'] = 0
        data['wind_direction'] = -1
        data['wind_speed_kt'] = None
        data['wind_gust_kt'] = None
        return data

    # Extract matched components
    direction, speed, gust, unit = match.groups()

    # Set direction to 999 if variable (VRB) or missing
    direction = -1 if direction == 'VRB' or direction is None else int(direction)
    speed = int(speed) if speed and speed.isdigit() else None
    gust = int(gust[1:]) if gust and gust[1:].isdigit() else None

    # Convert speeds to knots (KT) if necessary
    if unit == 'MPS':
        speed = round(speed * 1.94384) if speed else None  # MPS to KT
        gust = round(gust * 1.94384) if gust else None
    elif unit == 'KMH':
        speed = round(speed * 0.539957) if speed else None  # KMH to KT
        gust = round(gust * 0.539957) if gust else None

    # Update data dictionary with processed values
    data['has_wind'] = 1 if speed is not None else 0
    data['wind_is_variable'] = 1 if direction == -1  else 0
    data['wind_direction'] = direction
    data['wind_speed_kt'] = speed
    data['wind_gust_kt'] = gust
    return data

weather_mapping = {
    "DZ": 1,  # Drizzle
    "RA": 3,  # Rain
    "SN": 5,  # Snow
    "SG": 2,  # Snow Grains
    "IC": 2,  # Ice Crystals
    "PL": 3,  # Ice Pellets
    "GR": 6,  # Hail
    "GS": 4,  # Small Hail or Snow Pellets
    "UP": 2,  # Unknown Precipitation
    "BR": 1,  # Mist
    "FG": 3,  # Fog
    "FU": 4,  # Smoke
    "VA": 5,  # Volcanic Ash
    "DU": 3,  # Dust
    "SA": 4,  # Sand
    "HZ": 2,  # Haze
    "PY": 3,  # Spray
    "PO": 4,  # Dust/Sand Whirls
    "SQ": 5,  # Squall
    "FC": 6,  # Funnel Cloud (Tornado/Waterspout)
    "SS": 6,  # Sandstorm
    "DS": 6,  # Duststorm
    "SH": 4,  # Showers
    "TS": 5,  # Thunderstorm
    "FZ": 5   # Freezing
}

# Intensity multipliers
intensity_mapping = {
    "-": 0.5,  # Light intensity
    "+": 1.5,  # Heavy intensity
    None: 1.0  # Standard intensity
}

def process_weather(data):
    """Processes the 'weather' field in the data dictionary to assign a score based on intensity and type."""
    weather_str = data.get('weather', None)
    if not weather_str:
        data['weather_score'] = 0  # Default score if no weather data is present
        return data

    # Extract intensity and weather type
    match = re.match(r'^(\+|-)?([A-Z]{2,6})$', weather_str)
    if not match:
        data['weather_score'] = 0  # Default score for unrecognized patterns
        return data

    # Get the intensity and weather code
    intensity, weather_code = match.groups()

    # Calculate score based on weather code and intensity
    base_score = weather_mapping.get(weather_code, 0)  # Get base score for weather type
    multiplier = intensity_mapping.get(intensity, 1.0)  # Apply multiplier based on intensity
    score = base_score * multiplier

    # Update data dictionary with calculated weather score
    data['weather_score'] = score
    return data

def process_qnh(data):
    """
    Process the 'qnh' field to a unified unit in HPA.
    If the unit is in inches (INS), convert it to HPA.
    If the unit is missing, set it as None.
    """
    qnh_str = data.get('qnh', None)

    if qnh_str is None:
        data['qnh_hpa'] = None
        return data

    # Match QNH format
    match = re.match(r'QNH(\d{4})(INS|HPA)?', qnh_str)
    if match:
        qnh_value, unit = match.groups()
        qnh_value = int(qnh_value)

        # Convert INS to HPA if necessary
        if unit == "INS":
            qnh_value = round(qnh_value * 33.8639)  # 1 inHg = 33.8639 hPa
        data['qnh_hpa'] = qnh_value
    else:
        data['qnh_hpa'] = None  # Default for unrecognized patterns

    return data

def process_variable_wind(data):
    """
    Process the 'variable_wind' field to extract the variable wind directions.
    If the wind is variable, extract starting and ending directions in degrees.
    If the format is unrecognized or missing, set values to None.
    """
    variable_wind_str = data.get('variable_wind', None)

    if variable_wind_str is None:
        data['variable_wind_from'] = None
        data['variable_wind_to'] = None
        return data

    # Match variable wind format WND XXXVYYY
    match = re.match(r'WND\s+(\d{3})V(\d{3})', variable_wind_str)
    if match:
        from_dir, to_dir = match.groups()
        data['variable_wind_from'] = int(from_dir)
        data['variable_wind_to'] = int(to_dir)
    else:
        # If the format is not recognized, set to None
        data['variable_wind_from'] = None
        data['variable_wind_to'] = None

    return data

def convert_temp_to_datetime(data):
    """
    Converts max_temp_day/max_temp_hour and min_temp_day/min_temp_hour to full UTC datetime.
    Adjusts times set to 24:00 to 00:00 of the following day.
    """
    # Get base date from data['date_time']
    base_date = pd.to_datetime(data.get('date_time', None), utc=True)
    if base_date is None:
        data['max_temperature_time'] = None
        data['min_temperature_time'] = None
        return data

    # Helper function to handle day/hour adjustments
    def calculate_temperature_time(day, hour):
        # Check if day or hour is None; return None if either is missing
        if day is None or hour is None:
            return None

        # Ensure `day` and `hour` are integers
        day = int(day)
        hour = int(hour)

        # If hour is 24, reset it to 0 and increment day by 1
        hour %= 24
        day += hour //24

        # Handle month rollover and invalid day/hour values
        try:
            temp_time = base_date.replace(day=day, hour=hour)
        except ValueError:
            # If day exceeds the days in the month, roll over to the next month
            temp_time = base_date + relativedelta(months=1, day=1, hour=hour) + timedelta(days=day - 2)

        # Ensure temp_time is after base_date, handling month rollover if necessary
        while temp_time < base_date:
            temp_time += relativedelta(months=1)

        return temp_time

    # Convert max and min temperature times using the helper function
    max_temp_day = data.get('max_temp_day')
    max_temp_hour = data.get('max_temp_hour')
    min_temp_day = data.get('min_temp_day')
    min_temp_hour = data.get('min_temp_hour')

    # Assign converted datetime values to the data dictionary
    data['max_temperature_time'] = calculate_temperature_time(max_temp_day, max_temp_hour)
    data['min_temperature_time'] = calculate_temperature_time(min_temp_day, min_temp_hour)

    return data

def process_probability(data):
    prob_str = data.get('probability', None)

    if prob_str and 'PROB' in prob_str:
        data['probability'] = round(float(data['prob_value']) / 100, 2)
        data['has_prob'] = True
    else:
        data['probability'] = 0.9
        data['has_prob'] = False

    return data


def remove_duplicates_in_report(report):
    words = report.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)

# Parse TAF report
def parse_taf_block(date_time, line):
    """Parse a full TAF block into its components."""

    # Regex pattern to match main TAF components across multiple lines
    match = taf_pattern.match(line)

    if match:
        data = match.groupdict()

        # Convert `date_time` to UTC format
        try:
            utc_date = datetime.strptime(date_time, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
            data['date_time'] = utc_date.strftime("%Y-%m-%d %H:%M:%S")
            data['date_time'] = pd.to_datetime(data['date_time'], utc=True)
        except ValueError as e:
            return None

        # Amended handling
        data['amended'] = True if data.get('amended') == 'AMD' else False

        # corection handling
        data['corection'] = True if data.get('corection') == 'COR' else False

        # validity date handling
        data = process_validity(data)

        data = process_wind(data)

        visibility = data.get('visibility')
        if visibility == "CAVOK":
            data['visibility_meters'] = 10000  # Convention for CAVOK
        elif visibility and "SM" in visibility:

            visibility_miles = float(visibility.replace("SM", ""))
            data['visibility_meters'] = int(visibility_miles * 1609.34)
        elif visibility and visibility.isdigit():
            data['visibility_meters'] = int(visibility)
        else:
            data['visibility_meters'] = None

        max_temp_value = data.get('max_temp_value', None)
        if max_temp_value is not None:
            value = re.search(r'\d+', max_temp_value)
            data['max_temp_value'] = int(max_temp_value) if 'M' not in max_temp_value else 0 - int(value.group())

        min_temp_value = data.get('min_temp_value', None)
        if min_temp_value is not None:
            value = re.search(r'\d+', min_temp_value)
            data['min_temp_value'] = int(min_temp_value) if 'M' not in min_temp_value else 0 - int(value.group())

        data = process_weather(data)

        clouds = data.get('clouds')
        data['cloud_layers'] = parse_cloud_layers(clouds.strip().split()) if clouds else []

        data = process_qnh(data)

        data = process_variable_wind(data)

        data = convert_temp_to_datetime(data)
        data = process_probability(data)
        fields_to_drop = ['valid_start_day', 'valid_start_hour', 'valid_end_hour', 'valid_end_day',
                           'wind', 'clouds', 'qnh', 'variable_wind', 'max_temp_day', 'max_temp_hour'
                           , 'min_temp_day', 'min_temp_hour', 'max_temp', 'min_temp', 'validity',
                           'prob_value', 'visibility']
        for field in fields_to_drop:
            if field in data:
                del data[field]
        return data
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
        if data_type == "METAR":
            # METAR and TAF Paths: data/METAR/train/part_X/
            if dataset_purpose == "train" and not path_level:
                raise ValueError(f"{data_type} data requires path_level")
            if dataset_purpose == "train":
                path = os.path.join(base_dir, data_type, dataset_purpose, path_level)
            else:
                path = os.path.join(base_dir, data_type, dataset_purpose)
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
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines if line.strip()]
                data_entries = []
                date_time = None  # To keep track of the current date and time

                for line in lines:
                    # Check if the line is a date line
                    if re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}', line):
                        date_time = line  # Store the date and time
                        #print(date_time)
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
            current_date_time = None
            current_report_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                # Check if the line contains a date-time
                date_time_match = re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}', line)
                if date_time_match:
                    # If there’s an accumulated report, parse it
                    if current_report_lines:
                        full_report = ' '.join(current_report_lines)
                        # Split multiple TAF reports within the same block
                        taf_reports = re.split(r'=\s*', full_report)
                        for report in taf_reports:
                            report = report.strip()
                            if report:
                                report = remove_duplicates_in_report(report)
                                parsed_data = parse_taf_block(current_date_time, report)
                                if parsed_data:
                                    data_entries.append(parsed_data)
                        current_report_lines = []
                    current_date_time = date_time_match.group()
                else:
                    # Continue accumulating lines for the current report
                    current_report_lines.append(line)

            # Process the last report if any
            if current_report_lines:
                full_report = ' '.join(current_report_lines)
                taf_reports = re.split(r'=\s*', full_report)
                for report in taf_reports:
                    report = report.strip()
                    if report:
                        report = remove_duplicates_in_report(report)
                        parsed_data = parse_taf_block(current_date_time, report)
                        if parsed_data:
                            data_entries.append(parsed_data)
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
