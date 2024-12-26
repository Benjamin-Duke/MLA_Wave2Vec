# Assurez-vous d'être sur main
git checkout main

# Fusionner toutes les branches locales dans main
for branch in $(git branch --format='%(refname:short)' | grep -v "main"); do
  git merge "$branch" --no-ff -m "Merge $branch into main"
done

# Supprimer les branches locales fusionnées
git branch --merged main | grep -v "main" | xargs -n 1 git branch -d

# Supprimer les branches distantes fusionnées
for branch in $(git branch -r --merged main | grep -v "origin/main" | awk -F'/' '{print $2}'); do
  git push origin --delete "$branch"
done

