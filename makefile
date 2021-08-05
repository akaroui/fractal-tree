all : py git pip

py :
	python3 increaseVersionSetup.py
	rm -rf build dist
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload  dist/*

git :
	git commit -am 'update FractalTree package'
	git push origin master

pip :
	pip install --upgrade FractalTree
