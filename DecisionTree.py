import math 


# tree nodes 
#     attribute 
#     dictionary {attribute_value : node_pointer}
#     final_class : +, - 


# tree class 
#     max_depth
#     root - > node class 


class DecisionTree:
    class node:
        def __init__(self, attribute=None, final_class=None) -> None:
            self.attribute = attribute
            self.final_class = final_class
            self.next = {} 

    def __init__(self, max_depth=10) -> None:
        self.max_depth = max_depth 
    
    def fit(self, X_train, y_train):
        self.root = self.build(self, X_train, y_train, {})
    
    def entropy(self, X_train, y_train, attributes_taken):
        y_count = { key : 0 for key in y_train.unique() }

        for index, row in X_train.iterrows():
            correct_row = True 
            
            for attr, val in attributes_taken.items():
                if row[attr] != val:
                    correct_row = False
            
            if not correct_row:
                continue
            
            y_count[y_train[index]] += 1 
        
        tot = sum(value for value in y_count.values())
        
        # if any(value==sum for value in y_count.values()):
        #     for key in y_count:
        #         if y_count[key] == tot:
        #             return key 
        # else:
        entropy = 0.0 
        for value in y_count.values():
            if value == 0:
                continue 
            entropy -= (value/tot)*math.log2(value/tot)
        
        return entropy

    def __build_leaf(self, X_train, y_train, attributes_taken):
        # build leaf node here, check majority of current classes, 
        return 0
    
    
    def information_gain(self, X_train, y_train, attributes_taken, attr, initial_entropy):
        attr_values_cnt = { key : 0 for key in X_train[attr].unique() } 
        attr_values_entropies = { key : 0 for key in X_train[attr].unique() } 

        tot_cnt = 0 

        for attr_value in attr_values_cnt:
            attributes_taken[attr] = attr_value 

            attr_values_entropies[attr_value] = self.entropy(X_train, y_train, attributes_taken) 

            cnt = 0
            for index, row in X_train.iterrows():
                correct_row = True 

                for attribute_taken, value in attributes_taken.items():
                    if row[attribute_taken] != value:
                        correct_row = False
                
                if not correct_row:
                    continue

                #Correct row 
                cnt += 1 
            
            # store the Sv 
            attr_values_cnt[attr_value] = cnt 
            tot_cnt += cnt 
            
            # remove from attributes taken attr
            attributes_taken.pop(attr)
        
        new_entropy = 0 
        for attr_value in attr_values_cnt:
            new_entropy += (attr_values_cnt[attr_value]/tot_cnt) * attr_values_entropies[attr_value]

        # print(f"this is new entropy {new_entropy}")
        return initial_entropy - new_entropy 
               
    
    def build(self, X_train, y_train, attributes_taken):
        initial_entropy = self.entropy(X_train, y_train, attributes_taken)

        if initial_entropy == 0.0:
            #Make a leaf node 

            leaf = self.build_leaf(self, X_train, y_train, attributes_taken)
            return leaf 
        else:
            igs = {}
            for attr in X_train.columns:
                if attr in attributes_taken:
                    continue 

                igs[attr] = self.information_gain(X_train, y_train, attributes_taken, attr, initial_entropy) 
            
            print(igs)
            choosen_one = max(igs, key=igs.get) 

            print(f"We have choose {choosen_one} with IG {igs[choosen_one]}")
            # new_node = self.node(attribute=choosen_one)
            # move to childs 
        