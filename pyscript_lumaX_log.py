import sys
import requests
from datetime import datetime
from time import sleep

def main():
	ipaddr = "192.168.69.103"
	current_datetime = datetime.now()
	current_date_time = current_datetime.strftime("%Y%m%d-%H%M%S")
	print("CURR DATE ", current_date_time)

	with open("lumaX" + current_date_time + ".json", "w") as fp:
		true = 0
		while true == 0:
			response = requests.get("http://192.168.69.105/api/status.json")
			fp.write(str(response.json()))
			cd = datetime.now()
			cd1 = cd.strftime("%Y%m%d-%H%M%S")
			fp.write("\n")
			fp.write(str(cd1))
			fp.write("\n")
			print(str(cd1))
			sleep(1)
		fp.close()



if __name__ == "__main__":
	main()
	
