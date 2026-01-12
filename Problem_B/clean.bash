echo 'machine 1'
sshpass -p 991215 ssh yan@10.22.10.78 "ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9"
echo 'machine 2'
sshpass -p 991215 ssh yan@10.22.10.68 "ps aux | grep 'python' | grep -v grep | awk '{print $2}' | xargs kill -9"

echo 'machine 2'
sshpass -p 991215 ssh yan@10.22.10.84 "ps aux | grep 'python' | grep -v grep | awk '{print $2}' | xargs kill -9"


echo 'machine 2'
sshpass -p 991215 ssh yan@10.22.10.66 "ps aux | grep 'python' | grep -v grep | awk '{print $2}' | xargs kill -9"

