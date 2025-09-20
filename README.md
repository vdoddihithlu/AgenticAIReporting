conda create -p venv python==3.8 -y 
activate venv\

#set path when files not found on cmd
set PYTHONPATH=%PYTHONPATH%;C:\myCODE\AgenticAIReporting\AgenticAIReporting


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

#to revert the last commit 
git reset --soft d207c94^
commit & force push later
git push origin main --force

#####################################################################################
https://poloclub.github.io/transformer-explainer/