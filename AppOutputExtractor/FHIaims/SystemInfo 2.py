#

from datetime import datetime

def time_diff_in_seconds(time_str1, time_str2):
    # Convert the time strings to datetime objects
    dt1 = datetime.strptime(time_str1, "Date : %Y%m%d, Time : %H%M%S.%f")
    dt2 = datetime.strptime(time_str2, "Date : %Y%m%d, Time : %H%M%S.%f")

    # Calculate the time difference in seconds
    time_diff = abs((dt2 - dt1).total_seconds())

    # Return the time difference in seconds
    return time_diff


if __name__ == '__main__':

	time_str1 = "Date : 20230426, Time : 002047.673"
	time_str2 = "Date : 20230426, Time : 001834.244"

	time_diff = time_diff_in_seconds(time_str1, time_str2)

	print("Time difference in seconds:", time_diff)





