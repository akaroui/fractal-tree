import re

with open('setup.py', 'r') as f:
    file = f.read()

m = re.search('version="(.+?)"', file)
vs = m.group(1).split('.')
vs[-1] = str(int(vs[-1])+1).zfill(3)
vs = ".".join(vs)
modif = m.group(0).replace(m.group(1), vs)

with open('setup.py', 'w') as f:
    f.write(file.replace(m.group(0), modif))
