from extrap.entities.coordinate import Coordinate

def check_model_requirements(experiment, min_points):
    modeling_reuqirements_satisfied = False
    # one model parameter
    if len(experiment.parameters) == 1:
        if len(experiment.coordinates) >= min_points:
            modeling_reuqirements_satisfied = True

    # two model parameters
    elif len(experiment.parameters) == 2:
        x_requirement = False
        y_requirement = False
        
        # find the cheapest line of 5 points for y
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x:
                    y.append(cord_values2[1])
            if len(y) == 5:
                if x not in y_lines:
                    y_lines[x] = y
        if len(y_lines) >= 1:
            y_requirement = True
        #print("y_lines:",y_lines)

        # find the cheapest line of 5 points for x
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y:
                    x.append(cord_values2[0])
            if len(x) == 5:
                if y not in x_lines:
                    x_lines[y] = x
        if len(x_lines) >= 1:
            x_requirement = True
        #print("x_lines:",x_lines)

        if x_requirement and y_requirement:
            modeling_reuqirements_satisfied = True

    # three model parameters
    elif len(experiment.parameters) == 3:
        x_requirement = False
        y_requirement = False
        z_requirement = False

        # find the cheapest line of 5 points for y
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[2] == z:
                    y.append(cord_values2[1])
            if len(y) >= 5:
                if (x,z) not in y_lines:
                    y_lines[(x,z)] = y
        if len(y_lines) >= 1:
            y_requirement = True
        #print("y_lines:",y_lines)

        # find the cheapest line of 5 points for x
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y and cord_values2[2] == z:
                    x.append(cord_values2[0])
            if len(x) >= 5:
                if (y,z) not in x_lines:
                    x_lines[(y,z)] = x
        if len(x_lines) >= 1:
            x_requirement = True
        #print("x_lines:",x_lines)

        # find the cheapest line of 5 points for z
        z_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            z = []
            y = cord_values[1]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[1] == y:
                    z.append(cord_values2[2])
            if len(z) >= 5:
                if (x,y) not in z_lines:
                    z_lines[(x,y)] = z
        if len(z_lines) >= 1:
            z_requirement = True
        #print("z_lines:",z_lines)

        if x_requirement and y_requirement and z_requirement:
            modeling_reuqirements_satisfied = True

    # four model parameters
    elif len(experiment.parameters) == 4:
        x1_requirement = False
        x2_requirement = False
        x3_requirement = False
        x4_requirement = False

        # find the cheapest line of 5 points for x1
        x1_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = []
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == x2 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x1.append(cord_values2[0])
            if len(x1) >= 5:
                if (x2,x3,x4) not in x1_lines:
                    x1_lines[(x2,x3,x4)] = x1
        if len(x1_lines) >= 1:
            x1_requirement = True
        #print("x1_lines:",x1_lines)

        # find the cheapest line of 5 points for x2
        x2_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = []
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x2.append(cord_values2[1])
            if len(x2) >= 5:
                if (x1,x3,x4) not in x2_lines:
                    x2_lines[(x1,x3,x4)] = x2
        if len(x2_lines) >= 1:
            x2_requirement = True
        #print("x2_lines:",x2_lines)

        # find the cheapest line of 5 points for x3
        x3_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = []
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[3] == x4:
                    x3.append(cord_values2[2])
            if len(x3) >= 5:
                if (x1,x2,x4) not in x3_lines:
                    x3_lines[(x1,x2,x4)] = x3
        if len(x3_lines) >= 1:
            x3_requirement = True
        #print("x3_lines:",x3_lines)

        # find the cheapest line of 5 points for x4
        x4_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[2] == x3:
                    x4.append(cord_values2[3])
            if len(x4) >= 5:
                if (x1,x2,x3) not in x4_lines:
                    x4_lines[(x1,x2,x3)] = x4
        if len(x4_lines) >= 1:
            x4_requirement = True
        #print("x4_lines:",x4_lines)

        if x1_requirement and x2_requirement and x3_requirement and x4_requirement:
            modeling_reuqirements_satisfied = True

    return modeling_reuqirements_satisfied


def identify_selection_mode(experiment, min_points):
    selection_mode = None

    # check if there are enough measurement points for Extra-P to create a model
    modeling_reuqirements_satisfied = check_model_requirements(experiment, min_points) 

    # if modeling requirements are satisfied check if an additional point that is not part of the lines is existing already (only for 2+ parameters)
    if modeling_reuqirements_satisfied:
        if len(experiment.parameters) > 1:
            additional_point_exists = check_additional_point(experiment)
            print("DEBUG additional_point_exists:",additional_point_exists)
            # if additional point is available suggest more points using gpr method
            if additional_point_exists:
                selection_mode = "gpr"
            # if not available an additional point suggest points using add method
            else:
                selection_mode = "add"
        else:
            #TODO: or maybe later a special option if that does not work with GPR, since it was never testes before
            selection_mode = "gpr"
    # if modeling requirements are not satisfied suggest points using base method
    else:
        selection_mode = "base"
    
    return selection_mode


def check_additional_point(experiment):
    # two model parameters
    if len(experiment.parameters) == 2:
        # find the possible y lines
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x:
                    y.append(cord_values2[1])
            if len(y) == 5:
                if x not in y_lines:
                    y_lines[x] = y
        base_cords = []
        for key, value in y_lines.items():
            for val in value:
                base_cords.append(Coordinate((key,val)))
        # find the possible x lines
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y:
                    x.append(cord_values2[0])
            if len(x) == 5:
                if y not in x_lines:
                    x_lines[y] = x
        for key, value in x_lines.items():
            for val in value:
                if Coordinate((val,key)) not in base_cords:
                    base_cords.append(Coordinate((val,key)))
        additional_cord_found = False
        for i in range(len(experiment.coordinates)):
            if experiment.coordinates[i] not in base_cords:
                additional_cord_found = True
                break
        
        if len(x_lines) > 1 or len(y_lines) > 1:
            additional_cord_found = True
 
    # three model parameters
    elif len(experiment.parameters) == 3:
        # find the cheapest line of 5 points for y
        y_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            y = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[2] == z:
                    y.append(cord_values2[1])
            if len(y) >= 5:
                if (x,z) not in y_lines:
                    y_lines[(x,z)] = y
        base_cords = []
        for key, value in y_lines.items():
            for val in value:
                base_cords.append(Coordinate((key[0],val,key[1])))

        # find the cheapest line of 5 points for x
        x_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            y = cord_values[1]
            x = []
            z = cord_values[2]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == y and cord_values2[2] == z:
                    x.append(cord_values2[0])
            if len(x) >= 5:
                if (y,z) not in x_lines:
                    x_lines[(y,z)] = x
        for key, value in x_lines.items():
            for val in value:
                if Coordinate((val,key[0],key[1])) not in base_cords:
                    base_cords.append(Coordinate((val,key[0],key[1])))

        # find the cheapest line of 5 points for z
        z_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x = cord_values[0]
            z = []
            y = cord_values[1]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x and cord_values2[1] == y:
                    z.append(cord_values2[2])
            if len(z) >= 5:
                if (x,y) not in z_lines:
                    z_lines[(x,y)] = z
        for key, value in z_lines.items():
            for val in value:
                if Coordinate((key[0],key[1],val)) not in base_cords:
                    base_cords.append(Coordinate((key[0],key[1],val)))

        additional_cord_found = False
        for i in range(len(experiment.coordinates)):
            if experiment.coordinates[i] not in base_cords:
                additional_cord_found = True
                break
        if len(y_lines) > 1 or len(x_lines) > 1 or len(z_lines) > 1:
            additional_cord_found = True

    # four model parameters
    elif len(experiment.parameters) == 4:
        # find the cheapest line of 5 points for x1
        x1_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = []
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[1] == x2 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x1.append(cord_values2[0])
            if len(x1) >= 5:
                if (x2,x3,x4) not in x1_lines:
                    x1_lines[(x2,x3,x4)] = x1
        base_cords = []
        for key, value in x1_lines.items():
            for val in value:
                base_cords.append(Coordinate((val,key[0],key[1],key[2])))

        # find the cheapest line of 5 points for x2
        x2_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = []
            x3 = cord_values[2]
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[2] == x3 and cord_values2[3] == x4:
                    x2.append(cord_values2[1])
            if len(x2) >= 5:
                if (x1,x3,x4) not in x2_lines:
                    x2_lines[(x1,x3,x4)] = x2
        for key, value in x2_lines.items():
            for val in value:
                if Coordinate((key[0],val,key[1],key[2])) not in base_cords:
                    base_cords.append(Coordinate((key[0],val,key[1],key[2])))

        # find the cheapest line of 5 points for x3
        x3_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = []
            x4 = cord_values[3]
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[3] == x4:
                    x3.append(cord_values2[2])
            if len(x3) >= 5:
                if (x1,x2,x4) not in x3_lines:
                    x3_lines[(x1,x2,x4)] = x3
        for key, value in x3_lines.items():
            for val in value:
                if Coordinate((key[0],key[1],val,key[2])) not in base_cords:
                    base_cords.append(Coordinate((key[0],key[1],val,key[2])))

        # find the cheapest line of 5 points for x4
        x4_lines = {}
        for i in range(len(experiment.coordinates)):
            cord_values = experiment.coordinates[i].as_tuple()
            x1 = cord_values[0]
            x2 = cord_values[1]
            x3 = cord_values[2]
            x4 = []
            for j in range(len(experiment.coordinates)):
                cord_values2 = experiment.coordinates[j].as_tuple()
                if cord_values2[0] == x1 and cord_values2[1] == x2 and cord_values2[2] == x3:
                    x4.append(cord_values2[3])
            if len(x4) >= 5:
                if (x1,x2,x3) not in x4_lines:
                    x4_lines[(x1,x2,x3)] = x4
        for key, value in x4_lines.items():
            for val in value:
                if Coordinate((key[0],key[1],key[2],val)) not in base_cords:
                    base_cords.append(Coordinate((key[0],key[1],key[2],val)))

        additional_cord_found = False
        for i in range(len(experiment.coordinates)):
            if experiment.coordinates[i] not in base_cords:
                additional_cord_found = True
                break
        
        if len(x1_lines) > 1 or len(x2_lines) > 1 or len(x3_lines) > 1 or len(x4_lines) > 1:
            additional_cord_found = True

    return additional_cord_found