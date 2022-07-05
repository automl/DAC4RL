if [[ -f "results.csv" ]]; then
    echo "file exists"
    rm -rf "results.csv"
fi

echo "Agent, Env, Result" >> results.csv

for env in CARLAcrobotEnv CARLCartPoleEnv CARLLunarLanderEnv CARLMountainCarContinuousEnv CARLMountainCarEnv CARLPendulumEnv CARLAnt CARLFetch CARLGrasp; do 
    for agent in DDPG PPO SAC; do
        echo "Running agent $agent on environment $env"
        if python test_agent.py --agent $agent --env $env --steps 1000 --outdir ./tmp/agent_sanity/$env/
        then
            echo "$agent,$env,SUCESS" >> results.csv
        else
            echo "$agent,$env,FAILURE" >> results.csv
        fi
    done
done