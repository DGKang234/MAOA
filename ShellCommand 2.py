import os, subprocess

class shellcommand():

	def __init__(self,tarfile=None):
		# path to the tarfile 'ASCII' format
		if tarfile != None:
			try:
				os.path.exists(tarfile)
				self.tarfile = tarfile
				self.filecheck = True
			except:
				self.filecheck = False
		else:
			self.filecheck = False

	def set_tarfile(self,tarfile):
		try:
			os.path.exists(tarfile)
			self.tarfile = tarfile
			self.filecheck = True
			return self.filecheck
		except:
			self.filecheck = False
			return self.filecheck

	def check_tarfile(self):
		return self.filecheck
	def get_tarfile_path(self):
		return self.tarfile

	def pipe(self,*args):
		if len(args) == 2:
			return args[0] + ' | ' + args[1]
		else:
			return False

	def grep(self, pattern, *args, headtail=None,file=True):

		# defulat
		if file == True:
			ret = 'grep {}'.format(pattern) + ' ' + self.tarfile	# if None use initial setup
		elif file == False:
			ret = 'grep {}'.format(pattern) + ' '

		# grep optionals
		for item in args:
			ret = ret + ' ' + item

		if headtail == 'head':
			ret = ret + ' | ' + 'head -1'
		if headtail == 'tail':
			ret = ret + ' | ' + 'tail -1'
		return ret

	def awk(self,*token,delimiter=None):

		ret = 'awk \'{print '

		if delimiter == None:
			delim = '\",\"'
		else:
			delim = delimiter
		# 
		for offset, item in enumerate(token):
			if offset != (len(token)-1):
				ret = ret + '${}'.format(item) + delim
			else:
				ret = ret + '${}'.format(item)
		#
		ret = ret + '}\''

		return ret

	def execute(self,command):
		try:
			ret = subprocess.check_output(command,shell=True)
			ret = ret.decode('utf-8')[:-1]
		
			return ret
		except:
			print('Error in command : {}'.format(command))

	def show(self,command):
		print(command)



class shell_check():

	NotImplemented
