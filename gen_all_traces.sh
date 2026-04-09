for food in {0..10}; do
    for service in {0..10}; do
        python trace_request.py --food $food --service $service --rules heitor3.rules --port 5095
    done
done