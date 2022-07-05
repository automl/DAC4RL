if [[ -f "time_results.csv" ]]; then
    echo "file exists"
    rm -rf "time_results.csv"
fi

echo "Agent, Env, Result" >> results.csv

for env in CARLPendulumEnv; do 
    for agent in DDPG PPO SAC; do
        echo "Running agent $agent on environment $env"
        if python test_agent.py --agent $agent --env $env --outdir ./tmp/time_results/$env/
        then
            echo "$agent,$env,SUCESS" >> time_results.csv
        else
            echo "$agent,$env,FAILURE" >> time_results.csv
        fi
    done
done