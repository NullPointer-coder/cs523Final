class Timestamp:
    def __init__(self, year=None, month=None, day=None, hour=None, minute=None):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute

    def __repr__(self):
        return (f"Timestamp(year={self.year}, month={self.month}, day={self.day}, "
                f"hour={self.hour}, minute={self.minute})")
    
    def to_string(self, format="%Y-%m-%d %H:%M"):
        """Returns a formatted string representation of the timestamp. 
        If any part is None, it is omitted from the output."""
        
        # Build a dictionary for format fields
        fields = {
            "%Y": str(self.year) if self.year is not None else "",
            "%m": str(self.month).zfill(2) if self.month is not None else "",
            "%d": str(self.day).zfill(2) if self.day is not None else "",
            "%H": str(self.hour).zfill(2) if self.hour is not None else "",
            "%M": str(self.minute).zfill(2) if self.minute is not None else ""
        }
        
        # Replace the format placeholders with actual values or empty strings
        formatted = format
        for key, value in fields.items():
            formatted = formatted.replace(key, value)
        
        # Strip any extra spaces or redundant separators (e.g., "--" or " :")
        formatted = formatted.replace("  ", " ").replace("--", "-").replace(" :", "").strip()
        
        return formatted

if __name__ == "__main__":
    timestamp1 = Timestamp(year=2022, month=9, day=24, hour=15, minute=30)
    timestamp2 = Timestamp(year=2023, month=1)

    print(timestamp1)  
    print(timestamp2)  

    print(timestamp1.to_string())         
    print(timestamp2.to_string("%Y-%m"))