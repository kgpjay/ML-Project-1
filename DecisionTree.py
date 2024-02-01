import math 
import pandas as pd 


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
        self.root = self.build(X_train, y_train, {})
        print("Done training")

    def predict(self, X_test):
        y_pred = pd.Series(range(X_test.shape[0]), index=X_test.index)

        for index, row in X_test.iterrows():
            print(index)
            y_pred[index] = self.get_class(row, self.root)
        
        return y_pred

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
               
    def build_leaf(self, X_train, y_train, attributes_taken):
        # build leaf node here, check majority of current classes, 
        class_cnt = { key : 0 for key in y_train.unique() }
        for index, row in X_train.iterrows():
            correct_row = True 

            for attribute_taken, value in attributes_taken.items():
                if row[attribute_taken] != value:
                    correct_row = False
            
            if not correct_row:
                continue
            
            class_cnt[y_train[index]] += 1 
        
        # get key max value in class_cnt
        choosen_class = max(class_cnt, key=class_cnt.get)

        #create a node object 
        return self.node(final_class=choosen_class)
    
    def build(self, X_train, y_train, attributes_taken):
        initial_entropy = self.entropy(X_train, y_train, attributes_taken)

        if initial_entropy == 0.0:
            #Make a leaf node 
            leaf = self.build_leaf(X_train, y_train, attributes_taken)
            return leaf 
        else:
            igs = {}
            for attr in X_train.columns:
                if attr in attributes_taken:
                    continue 

                igs[attr] = self.information_gain(X_train, y_train, attributes_taken, attr, initial_entropy) 
            
            # print(igs)
            choosen_attr = max(igs, key=igs.get) 

            # print(f"We have choose {choosen_attr} with IG {igs[choosen_attr]}")
            # new_node = self.node(attribute=choosen_one)

            # Exhuasted all atributes 
            if len(igs) == 0:
                leaf = self.build_leaf(X_train, y_train, attributes_taken)
                return leaf 
            else:
                # move to childs 
                new_node = self.node(attribute=choosen_attr)
                new_node.next = { key : None for key in X_train[choosen_attr].unique() }

                for value in new_node.next:
                    attributes_taken[choosen_attr] = value
                    new_node.next[value] = self.build(X_train, y_train, attributes_taken)
                    attributes_taken.pop(choosen_attr) 
                
                return new_node 

    def get_class(self, row, current_node):
        # some thing wrong happened  
        if not current_node:
            return None 

        if current_node.final_class:
            return current_node.final_class 
        else:
            # print(current_node.attribute, row[current_node.attribute], row[current_node.attribute] in current_node.next)
            if not row[current_node.attribute] in current_node.next:
                return None 
            return self.get_class(row, current_node.next[row[current_node.attribute]])
    
    def accuracy(self, y_pred, y_test):
        same = 0
        diff = 0 
        for index in y_test.index:
            if y_pred[index] == y_test[index]:
                same += 1 
            else:
                diff += 1 
        return same/(same+diff) 