conda create -p venv python==3.8 -y 
activate venv\

#set path when files not found
set PYTHONPATH=%PYTHONPATH%;C:\myCODE\myML


git init 
git config --global user.name "vidya" 
git config --global user.email "myemail" 

git remote add origin https://github.com/vdoddihithlu/myML.git 
git remote -v 
git branch -M main #### rename current branch to main

git add . # . is all 
git commit -m "new comit" 
git push -u origin main 
git status