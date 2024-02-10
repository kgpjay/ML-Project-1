import networkx as nx
import matplotlib.pyplot as plt
import math 
import os
import pandas as pd 
import time 


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

    def __init__(self, max_depth=None) -> None:
        self.max_depth = max_depth 
        self.depth = 0 
    
    def fit(self, X_train, y_train):
        self.root = self.build(X_train, y_train, attributes_taken={}, current_depth=1)
        print(f"Done training \nDepth of tree = {self.depth}")

    def predict(self, X_test):
        y_pred = pd.Series(range(X_test.shape[0]), index=X_test.index)

        for index, row in X_test.iterrows():
            y_pred[index] = self.get_class(row, self.root)
        
        # Find the maximum occurring value in y_pred
        max_occurring_value = y_pred.mode()[0]  # Mode returns a Series, so we take the first value

        # Replace None values in y_pred with the maximum occurring value
        y_pred_filled = y_pred.fillna(max_occurring_value)

        return y_pred_filled

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

        # print(choosen_class)

        #create a node object 
        return self.node(final_class=choosen_class)
    
    def build(self, X_train, y_train, attributes_taken, current_depth):
        initial_entropy = self.entropy(X_train, y_train, attributes_taken)
        self.depth = max(self.depth, current_depth)   #update current depth of tree 

        if not self.max_depth is None and self.max_depth == current_depth:
            '''
                Prune the branch
                Get the majority class 
            ''' 
            leaf = self.build_leaf(X_train, y_train, attributes_taken)
            return leaf 

        elif initial_entropy == 0.0:
            #Make a leaf node 
            # print(attributes_taken, end=" ") 
            
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

            if len(igs) == 0:
                # Exhuasted all atributes 
                leaf = self.build_leaf(X_train, y_train, attributes_taken)
                return leaf 
            else:
                # move to childs 
                new_node = self.node(attribute=choosen_attr)
                new_node.next = { key : None for key in X_train[choosen_attr].unique() }

                for value in new_node.next:
                    attributes_taken[choosen_attr] = value
                    new_node.next[value] = self.build(X_train, y_train, attributes_taken, current_depth+1)
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
    
    def draw_tree(self, filepath: str):
        G = nx.DiGraph()

        def add_edges(node):
            for edge, child_node in node.next.items():
                G.add_edge(node.attribute if node.attribute is not None else node.final_class,
                        child_node.attribute if child_node.attribute is not None else child_node.final_class,
                        label=edge)
                add_edges(child_node)

        add_edges(self.root)

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Gnodesep=1 -Granksep=1 -Goverlap=false -Gsplines=line -Gmodel=subset -Gstrict=false -Grankdir=TB')

        # Customize node shape and size
        node_shape = 's'  # Square
        node_size = 700

        #clear the plot 
        plt.clf()        

        # Visualize the tree with straight edges and square nodes
        nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", node_shape=node_shape, arrowsize=20, edgecolors='black', linewidths=1, width=1)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        # save the plot
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)

    def count_total_nodes(self, node_ptr=None):
        if node_ptr is None:
            node_ptr = self.root 
        
        if node_ptr.attribute is None:
            return 1 
        else:
            count = 1 
            for next in node_ptr.next.values():
                count += self.count_total_nodes(next)
        
        return count 

    def reduced_err_prunning(self, X_val, y_val):

        #validation accuracy before pruning 
        val_accuracy_before_pruning = self.accuracy(self.predict(X_val), y_val)
        print(f"Accuracy on validation (before pruning) = {val_accuracy_before_pruning}")
        print(f"Total Nodes (before pruning) = {self.count_total_nodes()}")

        #pruning 
        self.prune_subtree_rec(self.root, X_val, y_val, X_val_stat=X_val, y_val_stat=y_val)

        #validation accuracy after pruning 
        val_accuracy_after_pruning = self.accuracy(self.predict(X_val), y_val)
        print(f"Accuracy on validation (after pruning) = {val_accuracy_after_pruning}")
        print(f"Total Nodes (after pruning) = {self.count_total_nodes()}")

    def prune_subtree_rec(self, node, X_val, y_val, X_val_stat, y_val_stat):
        
        if node.attribute is None:
            return 
        
        #prune children if not leaf node 
        for attr_value, child in node.next.items():
            X_val_sub = X_val[X_val[node.attribute] == attr_value ]
            y_val_sub = y_val.loc[X_val_sub.index] 

            if len(y_val_sub) == 0:
                continue
            self.prune_subtree_rec(child, X_val_sub, y_val_sub, X_val_stat, y_val_stat)

        val_accuracy_before_prunning = self.accuracy(self.predict(X_val_stat), y_val_stat)

        #prune the subtree rooted at this node 
        choosen_attribute = node.attribute 
        node.attribute = None 
        node.final_class =  y_val.value_counts().idxmax()

        val_accuracy_after_prunning = self.accuracy(self.predict(X_val_stat), y_val_stat)
        # if accuracy decreases revert back to original node 
        if val_accuracy_after_prunning < val_accuracy_before_prunning + 0.00001:
            node.attribute = choosen_attribute
            node.final_class = None 
        else:
            #keeping it as leaf node, clearning next pointers 
            next = {}
    
        