from predict import main

import os
import json

json_result = []
i = 0
for filename in os.listdir("val2014"):
    path = os.path.join("val2014", filename)
    result = main(path)
    i+=1
    my_path = "result/{}.json".format(filename)
    with open(my_path, "w") as f:
        json.dump(result, f, indent=4)



