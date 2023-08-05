#

import json

class BaseExtractor(object):

	def __init__(self,app=None,version=None):

		self.app = app
		self.app_version = version

	def load_patterns(self,path):

		try:
			with open('{}/{}_{}_patterns.json'.format(path,self.app,self.app_version),'r') as f:
				return json.load(f)

		except FileNotFoundError as e:
			print(e)

	def get_appinfo(self):

		print('App = {} / AppVersion = {}'.format(self.app,self.app_version))



if __name__ == '__main__':

	be = BaseExtractor()
	print(be.get_appinfo())
