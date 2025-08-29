import os
import subprocess

def main():
    path = os.path.join(os.path.dirname(__file__),"susc.ipynb")
    path = r"{} ".format(path)
    try:
        option = r'--Voila.tornado_settings={}'.format('''{'websocket_max_message_size': 209715200}''')
        subprocess.check_output(['voila',path,option])
    except subprocess.CalledProcessError:
        option =  " --Voila.tornado_settings \"'websocket_max_message_size'=209715200\""
        command = "voila "+path+option
        subprocess.run(command, shell=True)
