
# Needed Imports
import pandas as pd
import numpy as np
import networkx as nx
import warnings


# TODO: Better comments and organization
# TODO: Add verbose and counts to MDiN functions to deliver better reports of networks made
# TODO: Add possiblity of 'Fomula_MDiN' checking mult_edge_options to replace 'incoherent' edges instead of only removing the latter.
# TODO: Add extra args to MDiN functions to better allow choosing different parameters and options - partially done.

# Atomic masses - https://ciaaw.org/atomic-masses.htm
# Isotopic abundances- https://ciaaw.org/isotopic-abundances.htm/https://www.degruyter.com/view/journals/pac/88/3/article-p293.xml
# Isotopic abundances from Pure Appl. Chem. 2016; 88(3): 293–306,
# Isotopic compositions of the elements 2013 (IUPAC Technical Report), doi: 10.1515/pac-2015-0503
chemdict = {'H':(1.0078250322, 0.999844),
            'C':(12.000000000, 0.988922),
            'N':(14.003074004, 0.996337),
            'O':(15.994914619, 0.9976206),
            'Na':(22.98976928, 1.0),
            'P':(30.973761998, 1.0),
            'S':(31.972071174, 0.9504074),
            'Cl':(34.9688527, 0.757647),
            'F':(18.998403163, 1.0),
            'C13':(13.003354835, 0.011078) # Carbon 13 isotope
           } 


# Base Formula related functions
def formula_process(formula):
    """Transforms a formula in string format into a dictionary."""
    
    #results = pd.DataFrame(np.zeros((1,8)), columns = ['C','H','O','N','S','P','Cl','F'])
    # Empty dictionary to store the results
    results = dict.fromkeys(['C','H','O','N','S','P','Cl','F'], 0)
    count = ''
    letter = None
    minus = False
    
    # Run through the string
    for i in range(len(formula)):
        if formula[i].isupper(): # If i is an uppercase letter then it is an element
            if letter: # Just to account for the first letter in the formula where letter is None 
                # Reached another letter, store previous results and reset count
                if minus:
                    results[letter] = - int(count or 1)
                    minus = False
                else:
                    results[letter] = int(count or 1)
                count = ''
                
            if i+1 < len(formula): # In case it's a two letter element such as Cl
                if formula[i+1].islower(): # The second letter is always lower case
                    letter = formula[i] + formula[i+1] # Store new 2 letter element
                    continue
                    
            letter = formula[i] # Store new 1 letter element
            
        elif formula[i].isdigit():
            count = count + formula[i] # If number, add number to count
        
        elif formula[i] == '-':
            minus = True
    
    # Store results of the last letter
    if minus:
        results[letter] = - int(count or 1)
        minus = False
    else:
        results[letter] = int(count or 1)
                    
    return results
    
    
# Reading default list of chemical transformations
def default_transformation_groups():
    trans_groups = pd.read_csv('transgroups.csv', sep='\t').set_index('Label')
    comp = []
    for i in trans_groups.index:
        comp.append(formula_process(i))
    trans_groups['Comp.'] = comp
    #trans_groups['Comp.'].loc['O(-NH)']['N'] = -1
    #trans_groups['Comp.'].loc['O(-NH)']['H'] = -1
    #trans_groups['Comp.'].loc['NH3(-O)']['O'] = -1

    return trans_groups


def getmass(c,h,o,n,s,p,cl,f):
    "Get the exact mass for any formula with those 8 elements."
    massC = chemdict['C'][0] * c
    massH = chemdict['H'][0] * h
    massO = chemdict['O'][0] * o
    massN = chemdict['N'][0] * n
    massS = chemdict['S'][0] * s
    massP = chemdict['P'][0] * p
    massCl = chemdict['Cl'][0] * cl
    massF = chemdict['F'][0] * f

    massTotal = massC + massH + massO + massN + massS + massP + massCl + massF

    return massTotal


# Function related to obtaining a simple Mass-Difference Network
def simple_MDiN(node_list, trans_groups=None, ppm=1):
    """Build an MDiN based on a list of nodes and allowed chemical transformations.
    
       node_list: list of m/z peaks to build the network.
       trans_df: Pandas DataFrame with an index of the names of the chemical transformations and a columns with the name 'Mass'
    with the respective masses of the transformation. If None, a default list of 15 chemical transformations will be used.
       ppm: scalar, error in parts per million accepted to establish edges.
       
       return: Networkx Graph (MDiN) object, Transformations are stored as an edge attribute."""
    
    # Creating the graph and adding nodes
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    
    if trans_groups is None:
        #if trans_groups != None:
            #warnings.warn('Argument trans_groups passed is not a DataFrame. Using the default list instead.')
        trans_groups = default_transformation_groups()
    
    # For each node and for each transformation, see if adding the mass difference leads to another m/z on the node list 
    for node in node_list:
        for trans in trans_groups.index:
            theo = node + trans_groups['Mass'][trans] # 'Theoretical' mass by adding the chemical transformation
            err_marg = ppm/(10**6) * theo # Margin of error accepted for the observed m/z peak (depends on ppm and theoretical mass)
            
            # Select possible candidates for establishing edges
            opt_temp = np.array(graph.nodes())[np.array(graph.nodes()) < theo+err_marg]
            options = opt_temp[opt_temp > theo-err_marg]
            if len(options) == 1:
                graph.add_edge(node, options[0], Transformation = trans)
            if len(options) > 1:
                for i in options: # Multiple edges from the same node with the same transformation might be established
                    graph.add_edge(node, i, Transformation = trans)


    return graph


# Functions related to obtaining a 'univocal' Mass-Difference Network, that is an MDiN that doesn't use the same chemical 
# transformation in the same 'direction' (adding or subtracting chemical group) from one node.
def univocal_MDiN(node_list, trans_groups=None, ppm=1):
    """Builds a Mass-Difference Network taking into account provided chemical transformations and limiting edges to a maximum
    of 1 (in each 'direction') for each transformation/node combo.
    
       node_list: list of m/z peaks to build the network.
       trans_groups: Pandas DataFrame with an index of the names of the chemical transformations and a columns with the name 
    'Mass' with the respective masses of the transformation. Default is a list of 15 chemical transformations will be used.
       ppm: scalar, error in parts per million accepted to establish edges.
       
       return: Networkx Graph (MDiN) object, Transformations are stored as an edge attribute."""
    
    # Setting up the graph
    graph = nx.Graph()
    graph.add_nodes_from(node_list)

    if trans_groups is None:
        #if trans_groups != None:
            #warnings.warn('Argument trans_groups passed is not a DataFrame. Using the default list instead.')
        trans_groups = default_transformation_groups()
    
    # Building a df with all possible edges (node_dfs) and a dict including the error associated with each edge (poss_nodes)
    poss_nodes, node_dfs = all_possible_edges(node_list, trans_groups, ppm=ppm)
    
    # Establishing edges that only use the chemical transformation (in a given direction) once for each of the nodes
    # Storing edges that have multiple options for 'beginning' and/or 'ending' nodes in a dict (mult_edge_options)
    # i.e. From one node, multiple nodes are options for the same chemical transformation (either adding or subtracting)
    mult_edge_options = MDiN_builder_I(node_dfs, poss_nodes,  graph, trans_groups)
    
    # From the edges that have multiple possibilities for 'beginning' and/or 'ending' nodes, select the best option based
    # on the error associated with the edge
    MDiN_builder_II(mult_edge_options, graph)
    
    # Return the MDiN
    return graph


def all_possible_edges(node_list, trans_groups=None, ppm=1):
    "Secondary function of univocal_MDiN - build a df with all possible edges and a dict including the associated error."

    # Setting up the storage for results
    poss_nodes = {}
    node_dfs = {}
    
    if trans_groups is None:
        #if trans_groups != None:
            #warnings.warn('Argument trans_groups passed is not a DataFrame. Using the default list instead.')
        trans_groups = default_transformation_groups()

    # For each node
    for node in node_list:
        temp_dict = {}
        node_dfs[node] = {}
        # And each tansformation
        for trans in trans_groups.index:
            # Find possible m/z peak options for adding the mass change of the transformation to the m/z of the node
            theo = node + trans_groups['Mass'][trans]
            err_marg = ppm/(10**6) * theo

            opt_temp = np.array(node_list)[np.array(node_list) < theo+err_marg]
            options = opt_temp[opt_temp > theo-err_marg] # List of m/z options within the allowed error margin

            error_dicts = {}  
            
            # Saving possible edges in node_dfs and error associated with them on temp_dict to store in poss_nodes after
            if len(options) > 1: # More than 1 option
                error = np.abs(options - theo)/theo * 10**6
                #options = [(x,y) for y, x in sorted(zip(error, options), key=lambda pair: pair[0])]
                for i in range(len(options)):
                    error_dicts[options[i]] = error[i] # Store
                    node_dfs[node][options[i]] = trans
                temp_dict[trans] = error_dicts
                

            elif len(options) == 1: # Only 1 option
                error = np.abs(options - theo)/theo * 10**6
                error_dicts[options[0]] = error[0]
                node_dfs[node][options[0]] = trans  
                temp_dict[trans] = error_dicts

        if len(temp_dict) > 0:
            poss_nodes[node] = temp_dict
    
    # Transforming node_dfs from a dict of dicts into a DataFrame. Dropping all empty columns
    node_dfs = pd.DataFrame.from_dict(node_dfs).dropna(axis=1, how ='all')
            
    return poss_nodes, node_dfs


def MDiN_builder_I(node_dfs, poss_nodes, graph, trans_groups=None):
    """Secondary function of univocal_MDiN - make edges that only use the chemical transformation (in a given direction) once
    for each of the nodes. Storing edges that have multiple options for 'beginning' and/or 'ending' nodes i.e. from one node,
    multiple nodes are options for the same chemical transformation (either adding or subtracting).
    """
    
    mult_edge_options = {}
    # For each chemical transformation and node
    for trans in trans_groups.index:
        for node in node_dfs.columns:
            # If a node + chemical transformation combo only has one option for 'receiving node'
            if sum(node_dfs[node] == trans) == 1:
                receive_node = node_dfs.index[node_dfs[node] == trans][0]
                # And if only the original node leads to that receiving node with the specific chemical transformation
                if node_dfs.loc[receive_node].value_counts()[trans] == 1:
                    #print(node, trans, receive_node)
                    # Add an edge to the graph
                    graph.add_edge(node, receive_node, Transformation = trans, 
                                error = poss_nodes[node][trans][receive_node])
                
                # If more than the original node can lead to the receiving node with the specific chemical transformation
                # Store in the multiple options dictionary.
                else:
                    mult_edge_options[(node, receive_node, trans)] = poss_nodes[node][trans][receive_node]
            
            # If a node + chemical transformation combo has multiple options for 'receiving node'
            # Store all options in the multiple options dictionary
            elif sum(node_dfs[node] == trans) > 1:
                for i in node_dfs[node].loc[node_dfs[node] == trans].index:
                    mult_edge_options[(node, i, trans)] = poss_nodes[node][trans][i]
    
    # Sort the multiple options dictionary from the lowest associated edge error to the highest
    mult_edge_options = dict(sorted(mult_edge_options.items(), key=lambda item: item[1]))
    
    return mult_edge_options


def MDiN_builder_II(mult_edge_options, graph):
    """Secondary function of univocal_MDiN - select the best edges based on their associated errors from the multiple edge
    options dictionary."""
    
    completed_edges = []
    for (node1, node2, trans) in mult_edge_options.keys(): # From the lowest to the highest errors
        if len(completed_edges) == 0:
            # Add the edge with the lowest associated error and store it the completed edges list.
            graph.add_edge(node1, node2, Transformation = trans, error = mult_edge_options[(node1, node2, trans)])
            completed_edges.append((node1, node2, trans))
        else:
            #edge = True
            #for cnodes in completed_nodes:
            #    if sum(np.array((node1, node2, trans)) == np.array(cnodes)) >= 2:
            #        edge = False
            #        print(node1,node2,trans, cnodes)
            #if edge:

            # If there is any edge in the completed edges that have 2 of the initial node, ending node or chemical
            # transformation, then, a better alternative of the current edge has already been made
            if any([sum(np.array((node1, node2, trans)) == np.array(cedges)) >= 2 for cedges in completed_edges]):
                continue
            # Else, add the edge to the graph and to the completed edge list
            else:
                graph.add_edge(node1, node2, Transformation = trans, error = mult_edge_options[(node1, node2, trans)])
                completed_edges.append((node1, node2, trans))



# Functions relatd to building a 'Formula' Mass-Difference Network, that is, an MDiN built taking into account a list of m/z peaks with
# 'trusted'  formulas that lead to the assignment of formulas to other m/z peaks and that have 'internal consistency' in the network,
# aka, that formulas do not enter into conflict with each other based on the 'chemical transformation edges'.
def chem_trans(met_form, trans_form, r_type='addition'):
    """Gives a formula (dict form) based on a starting formula and an associated chemical transformation (addition/subtraction).
    
       met_form: dictionary; starting formula (dictionary form).
       trans_form: dictionary; chemical transformation to affect the starting formula (dictionary form).
       r_type: 'addition' or 'subtraction', whether the chemical transformation leads to the addition or removal of a chemical
    group, respectively.
       
       return: Dictionary, formula obtained from the chemical transformation on the initial formula."""
    
    formula = met_form.copy()
    if r_type == 'addition':
        for i in trans_form:
            formula[i] = met_form[i] + trans_form[i]
    elif r_type == 'subtraction':
        for i in trans_form:
            formula[i] = met_form[i] - trans_form[i]
    else:
        raise ValueError('Reaction Type (r_type) not recognized. Available: "addition", "subtraction".')
    return formula


def new_metabolite_info(graph, node, edge, trans_groups, r_type):
    "Gives metabolite information for the identification of node - formula, theoretical mass and error in ppm's."
    
    start_form = graph.nodes()[node]['Formula'] # Formula of starting node
    trans_group = trans_groups['Comp.'].loc[graph.edges()[edge]['Transformation']] # Chemical Transformation
    new_form = chem_trans(start_form, trans_group, r_type) # Formula obtained of ending node
    theo_mass = getmass(new_form['C'],new_form['H'],new_form['O'],new_form['N'],
                        new_form['S'],new_form['P'],new_form['Cl'],new_form['F']) # Theoretical mass of new formula
    form_error = np.abs((edge[1]-theo_mass)/theo_mass*(10**6)) # Error associated
    
    return new_form, theo_mass, form_error


def edge_remover(edges_to_remove, graph):
    "Removes edges stored in a list from a graph."
    for edge in set(edges_to_remove):
        if edge in graph.edges():
            graph.remove_edge(*edge)


common_range = {'H/C':(0.2,3.1),'N/C':(0,1.3),'O/C':(0,1.2),'P/C':(0,0.3),'S/C':(0,0.8),'F/C':(0,1.5), 'Cl/C':(0,0.8), 'P/O': (0,0.34)} 

def elemental_ratios_check(c, h, o, n, s, p, cl, f, ratios=common_range, valency=True, nops= True):
    """Performs different checks to a formula as a measure of its probability of existing in biological systems.
    
       c,h,o,n,s,p,cl,f: integers; number of C, H, O, N, S, P, Cl and F atoms respectively in the considered formula.
       ratios: dict; dictionary with (minimum and maximum) elemental ratios to consider for check with the following keys: 
    ['H/C', 'N/C', 'O/C', 'P/C', 'S/C', 'F/C', 'Cl/C', 'P/O']
       valency: bool (default: True); If True, performs Valency check (Lewis_Senior_rules function), if False, it skips this check.
       nops: bool (default: True); If True, performs NOPS heuristic probability check (NOPS function), if False, it skips this check.

       return: bool; True if formula follows the different checks employed, False if not.
    """
    # See checks employed in Kind and Fiehn, 2007.
    ratio_check = False
    if c > 0: # More than 1 carbon atom
        # Elemental Ratios Check
        if ratios['H/C'][0] <= (h/c) <= ratios['H/C'][1]:
            if ratios['O/C'][0] <= (o/c) <= ratios['O/C'][1]:
                if ratios['N/C'][0] <= (n/c) <= ratios['N/C'][1]:
                    if ratios['P/C'][0] <= (p/c) <= ratios['P/C'][1]:
                        if ratios['S/C'][0] <= (s/c) <= ratios['S/C'][1]:
                            if ratios['F/C'][0] <= (f/c) <= ratios['F/C'][1]:
                                if ratios['Cl/C'][0] <= (cl/c) <= ratios['Cl/C'][1]:
                                    if o != 0:
                                        if ratios['P/O'][0] <= (p/o) < ratios['P/O'][1]:
                                            PO_Ratio = True
                                        else:
                                            PO_Ratio = False
                                    else:
                                        PO_Ratio = True
                                    if PO_Ratio:
                                        # NOPS heuristic probability check
                                        if nops:    
                                            NOPS_ratio = NOPS_check(n,o,p,s)
                                        else:
                                            NOPS_ratio = True
                                        if NOPS_ratio:
                                            # (Maximum) Valency check
                                            if valency:
                                                Val_check,_ = Lewis_Senior_rules(c,h,o,n,s,p,cl,f)
                                                if Val_check:
                                                    ratio_check = True
                                            else:
                                                ratio_check = True
                                            

    return ratio_check

def NOPS_check(n,o,p,s):
    """Checks if the element counts follow the HNOPS heuristic probablility checks as delineated by the Kind and Fiehn, 2007.
    
       n,o,p,s: integers; number of N, O, P and S atoms respectively in the considered formula.
       
       returns: bool; True if it fulfills the conditions, False if it doesn't."""
    
    NOPS_ratio = True
    # Check each of the rules
    if (n > 1) and (o > 1) and (p > 1) and (s > 1): # NOPS
        if (n < 10) and (o < 20) and (p < 4) and (s < 3):
            NOPS_ratio = True 
        else:
            NOPS_ratio = False
    elif (n > 3) and (o > 3) and (p > 3): # NOP
        if (n < 11) and (o < 22) and (p < 6):
            NOPS_ratio = True
        else:
            NOPS_ratio = False
    elif (o > 1) and (p > 1) and (s > 1): # OPS
        if (o < 14) and (p < 3) and (s < 3):
            NOPS_ratio = True
        else:
            NOPS_ratio = False
    elif (n > 1) and (p > 1) and (s > 1): # PSN
        if (n < 10) and (p < 4) and (s < 3):
            NOPS_ratio = True
        else:
            NOPS_ratio = False
    elif (n > 6) and (o > 6) and (s > 6): # NOS
        if (n < 19) and (o < 14) and (s < 8):
            NOPS_ratio = True
        else:
            NOPS_ratio = False

    return NOPS_ratio

def Lewis_Senior_rules(c,h,o=0,n=0,s=0,p=0,cl=0,f=0):
    """See if the formula follows Lewis' and Senior's rules (considering all max possible valency states for each element
    except Cl).
    
       c,h,o,n,s,p,cl,f: integers; number of C, H, O, N, S, P, Cl and F atoms respectively in the considered formula.
       
       returns: (bool, bool); (considering max valency of each element, considering normal valency of each element), True if
    it fulfills the conditions, False if it doesn't."""
    
    # Normal_Valencies, Max_Valencies (only 1 value when it is the same) - Absolute values of the valencies
    valC = 4
    valH = 1 # Odd
    valO = 2
    valN, max_valN = 3, 5 # Odd
    valS, max_valS = 2, 6 
    valP, max_valP = 3, 5 # Odd
    valCl, max_valCl = 1, 7 # Odd 
    valF = 1 # Odd
    
    Valency = False
    Valency_normal = False
    # 1st rule - The sum of valences or the total number of atoms having odd valences is even.
    if (h + n + p + cl + f) % 2 == 0:
        # Elements with their max valences
        total_max_v = (valC * c) + (valH * h) + (valO * o) + (max_valN * n) + (max_valS * s) + (max_valP * p) + (
            max_valCl * cl) + (valF * f)
            
        # 2nd rule - The sum of valences is greater than or equal to twice the maximum valence.
        # Ok, this one sonly eliminates small molecules either way and we are searching for molecules with more than 100 Da.

        # 3rd rule - The sum of valences is greater than or equal to twice the number of atoms minus 1.
        natoms = c + h + o + n + s + p + cl + f
        if total_max_v >= (2*(natoms-1)):
            Valency = True
        
        # If the formula follows the rules with the elements with their Maximum Valencies, see if it follows with normal valencies
        if Valency:
            # Elements with their common valences
            total_v = (valC * c) + (valH * h) + (valO * o) + (valN * n) + (valS * s) + (valP * p) + (valCl * cl) + (valF * f)
            # 3rd rule - The sum of valences is greater than or equal to twice the number of atoms minus 1.
            #natoms = c + h + o + n + s + p + cl + f
            if total_v >= (2*(natoms-1)):
                Valency_normal = True

    return Valency, Valency_normal

def formula_attributer(node_list, graph, trans_groups, expansion_nodes, edges_to_remove, case4s, formula_fail_edges):
    """Based on a list of nodes with assigned formulas, assigns formulas to its neighboring nodes in a Mass-Difference Network
    based on the transformation attribute of the edges (checks viability and error associated of those formulas, removing the edge if
    coherency is impossible)."""
    
    for node in node_list:
            for edge in graph.edges(node): # Passing through each edge of each node

                # Define reaction types
                if edge [1] > edge[0]:
                    r_type = 'addition'
                else:
                    r_type = 'subtraction'

                # Getting the info for the metabolite to be identified - formula, theoretical mass and error in ppm's.
                new_form, theo_mass, form_error = new_metabolite_info(graph, node, edge, trans_groups, r_type)

                # See if the new formula fits the known chemical space (elemental ratios) of metabolites - wide
                # Currently, no Valency check is made
                ratio_check = elemental_ratios_check(new_form['C'],new_form['H'],new_form['O'],new_form['N'],
                                new_form['S'],new_form['P'],new_form['Cl'],new_form['F'])
                # If it doesn't fit the known chemical space
                if ratio_check == False:
                    # If the original node has a trusted formula (won't be changed in the future), break the edge.
                    if graph.nodes()[edge[0]]['resp_edge'] is None:
                        #removed_edges = removed_edges + 1
                        edges_to_remove.append(edge[:2])
                        continue
                    
                    # If the original node might have the formula overwritten, keep a note in the formula_fail_edges of the
                    # edge in the 'opposite direction' as to see if it fails to overwrite this formula to break the edge.
                    # If the 'opposite direction' never happens, no other source of formulas in that case, edges will be broken.
                    else:
                        formula_fail_edges.append((edge[1], edge[0]))
                        continue
                
                # If the new node has no formula assigned before, give it the new metabolite information.
                if graph.nodes[edge[1]]['Formula'] is None:
                    graph.nodes[edge[1]]['Formula'] = new_form
                    graph.nodes()[edge[1]]['Form_theo_mass'] = theo_mass
                    graph.nodes()[edge[1]]['Form_error'] = form_error
                    graph.nodes()[edge[1]]['resp_edge'] = [edge[0]] # Edge responsible for the attribution
                    expansion_nodes.append(edge[1]) # Add the new node to continue formula attribution in the next cycle.
                
                # If the new node already had an original formula assigned
                else:
                    # If the formula is the same as the new formula proposed
                    if graph.nodes[edge[1]]['Formula'] == new_form:
                        if graph.nodes()[edge[1]]['resp_edge'] is not None: # And it is not one of the initial (trusted) formulas
                            graph.nodes()[edge[1]]['resp_edge'].append(edge[0]) # Add this edge as another responsible for the formula.
                    
                    # If proposed formula is different
                    else:
                        # If the original formula is one of the initial trusted formulas, store edge to remove
                        if graph.nodes()[edge[1]]['resp_edge'] is None:
                            #removed_edges = removed_edges + 1
                            edges_to_remove.append(edge[:2])

                        # If the original formula has a lower associated error than the proposed formula
                        elif graph.nodes[edge[1]]['Form_error'] < form_error:
                            # And the starting node has one of the initial formulas, store edge to remove
                            if graph.nodes()[edge[0]]['resp_edge'] is None:
                                #removed_edges = removed_edges + 1
                                edges_to_remove.append(edge[:2])

                            # If the starting node formula is not one of the initial trusted formulas
                            else:
                                if r_type == 'subtraction':
                                    r_type = 'addition'
                                else:
                                    r_type = 'subtraction'
                                    
                                # See and get the info of the formula of the current node if starting with the target node formula.
                                alt_form, alt_theo_mass, alt_form_error = new_metabolite_info(
                                    graph, edge[1], (edge[1],node), trans_groups, r_type)
    
    #### I maybe should only remove this later
                                # If the error associated with this alternative formula is higher than that of the current node.
                                if alt_form_error > graph.nodes[node]['Form_error']:
                                    # print('Case 1 -  Break the connection')
                                    #removed_edges = removed_edges + 1
                                    edges_to_remove.append(edge[:2])
                            
                                elif edge in formula_fail_edges:
                                    # print("Case 2 (Special) - Formula in the other way doesn't make sense, so no overwrite")
                                    #removed_edges = removed_edges + 1
                                    edges_to_remove.append(edge[:2])
    #####                             
                                # If it is lower, prepare the overwrite protocol for the current node
                                else:
                                    # Test if the alternative formula fits the know chemical space
                                    alt_ratio_check = elemental_ratios_check(alt_form['C'],alt_form['H'],alt_form['O'],
                                            alt_form['N'],alt_form['S'],alt_form['P'],alt_form['Cl'],alt_form['F'])
                                    # If yes prepare the overwrite protocol by adding the current node to the expansion list
                                    # This will keep the node in the expansion list until the formula overwrite happens naturally
                                    if alt_ratio_check:
                                        # print('Case 2a - Overwrite Protocol')
                                        expansion_nodes.append(node)
                                        break # No need to keep formula assigning since this node will be overwritten
                                    # else:
                                        # print('Case 2b - Alt Formula Fails')
                                        #removed_edges = removed_edges + 1
                                        #edges_to_remove.append(edge[:2])
                                        # Edge will be removed in the end if needed

                        # If the proposed formula error is lower than the original formula
                        elif graph.nodes[edge[1]]['Form_error'] > form_error:
                            
                            if r_type == 'subtraction':
                                r_type = 'addition'
                            else:
                                r_type = 'subtraction'
                            # See and get the info of the formula of the current node if starting with the target node formula.
                            alt_form, alt_theo_mass, alt_form_error = new_metabolite_info(
                                    graph, edge[1], (edge[1],node), trans_groups, r_type)
                            
                            # If the alternative formula for the current node also has higher error
                            if alt_form_error > graph.nodes[node]['Form_error']:
                                # print('Case 3 - Overwriting Formula')
                                graph.nodes[edge[1]]['Formula'] = new_form
                                graph.nodes()[edge[1]]['Form_theo_mass'] = theo_mass
                                graph.nodes()[edge[1]]['Form_error'] = form_error
                                graph.nodes()[edge[1]]['resp_edge'] = [edge[0]]
                                expansion_nodes.append(edge[1])
                            
                            # If the alternative formula for the current node has a lower error - Complex and rare
                            # To solve this conundrum about choosing the best string of formulas, the one with the lowest
                            # average error between the 2 possible ways.
                            # To avoid possible endless loops, case4s list is made to stop if it keeps repeating.
                            else:
                                #print('Case 4 - The (More) Complex One')
                                if case4s.count(edge[1]) < 5:
                                    av_error_current = np.mean((form_error, graph.nodes[node]['Form_error']))
                                    av_error_alt = np.mean((graph.nodes[edge[1]]['Form_error'], alt_form_error))
                                    # Other way will happen naturally
                                    if av_error_current < av_error_alt:
                                        graph.nodes[edge[1]]['Formula'] = new_form
                                        graph.nodes()[edge[1]]['Form_theo_mass'] = theo_mass
                                        graph.nodes()[edge[1]]['Form_error'] = form_error
                                        graph.nodes()[edge[1]]['resp_edge'] = [edge[0]]
                                        expansion_nodes.append(edge[1])
                                        case4s.append(edge[1])


    #print('----')
    return graph, expansion_nodes, edges_to_remove, case4s, formula_fail_edges

def formula_MDiN(node_list, name_df, trans_groups=None, ppm=1):
    """Builds a Mass-Difference Network taking into account provided chemical transformations, limiting edges to a maximum of 1
    (in each 'direction') for each transformation/node combo and that has internal formula consistency starting from a list of
    nodes with 'trusted' formulas (also formula assignment).
    
       node_list: list of m/z peaks to build the network.
       name_df: pandas DataFrame; DataFrame with m/z peaks with 'trusted' formulas as index and with a column named 'Formula' with
    the mentioned formulas in string or dict format.
       trans_groups: pandas DataFrame with an index of the names of the chemical transformations and a columns with the name 
    'Mass' with the respective masses of the transformation. Default is alist of 15 chemical transformations will be used.
       ppm: scalar, error in parts per million accepted to establish edges.
       
       return: Networkx Graph (MDiN) object, Transformations are stored as an edge attribute, formulas, theoretical masses and
    associated errors are stored as node attributes."""

    if trans_groups is None:
        #if trans_groups != None:
            #warnings.warn('Argument trans_groups passed is not a DataFrame. Using the default list instead.')
        trans_groups = default_transformation_groups()

    # Building the univocal MDiN network for later formula assignment
    graph = univocal_MDiN(node_list, trans_groups=trans_groups, ppm=ppm)
    
    # Setting up initial extra node attributes and initial (trusted) formulas from name_df
    for node in graph.nodes():
        graph.nodes()[node]['Formula'] = None
        graph.nodes()[node]['Form_theo_mass'] = 0
        graph.nodes()[node]['Form_error'] = 0
        graph.nodes()[node]['resp_edge'] = None
        graph.nodes()[node]['Comp_Class'] = None
    for node in name_df.index:
        if isinstance(name_df.loc[node,'Formula'], str):
            form = formula_process(name_df.loc[node,'Formula'])
        elif isinstance(name_df.loc[node,'Formula'], dict):
            form = name_df.loc[node,'Formula'].isdict()
        else:
            raise ValueError("Formula format not recognized. Formula formats recognized example: 'C6H12O6', {'C':6, 'H':12, 'O':6}.")
        graph.nodes()[node]['Formula'] = form # Formula
        theo_mass = getmass(form['C'],form['H'],form['O'],form['N'],form['S'],form['P'],form['Cl'],form['F'])
        graph.nodes()[node]['Form_theo_mass'] = theo_mass # Formula Theoretical Mass
        form_error = np.abs((node-theo_mass)/theo_mass*(10**6))
        graph.nodes()[node]['Form_error'] = form_error # m/z node error in ppm's in comparison to theoretical mass
        
    # First 'loop' assigning formulas from the initial trusted formulas
    expansion_nodes = [] # Setting up important lists
    edges_to_remove = []
    formula_fail_edges = []
    case4s = []
    # First 'loop'
    graph, expansion_nodes, edges_to_remove, case4s, formula_fail_edges = formula_attributer(
        name_df.index, graph, trans_groups, expansion_nodes, edges_to_remove, case4s, formula_fail_edges)
    # Removing edge that make coherency impossible in the network
    edge_remover(edges_to_remove, graph)
    
    # All further loops for formula assignment
    edges_to_remove = [] # Re-setting edges_to remove
    while len(expansion_nodes) > 0:
        new_nodes = expansion_nodes
        expansion_nodes = [] # Re-setting expansion_nodes in each loop. Expansion nodes are, barring some unique exceptions,
        # nodes where a formula was assigned or changed in the last loop. When no new formula is assigned, it stops
        graph, expansion_nodes, edges_to_remove, case4s, formula_fail_edges = formula_attributer(
            new_nodes, graph, trans_groups, expansion_nodes, edges_to_remove, case4s, formula_fail_edges)
        #print(len(expansion_nodes))
    
    # Observing if the formula_fail_edges in the list are to be removed
    # No fast way to explain why it is done this way
    # If edge[0], edge[1] direction didn't work for formula propagation, see in the other direction.
    # If they also don't work (which they shouldn't), this will be added to edges_to_remove
    for extra_edges in formula_fail_edges.copy():
        #print(extra_edges)
        # If, in the other 'direction', the starting node never got a formula, then we remove the edge right away
        if graph.nodes()[extra_edges[0]]['Formula'] == None: 
            graph.remove_edge(*extra_edges)
        else:
            graph, expansion_nodes, edges_to_remove, case4s, _ = formula_attributer(
                [extra_edges[0]], graph, trans_groups, expansion_nodes, edges_to_remove, case4s, formula_fail_edges)
    
    # Just a precaution if some edges_to_remove might actually not be to remove due to posterior formula overwriting.
    # Some tests and overall thought must be made to see if keeping this is neccessary.
    last_nodes = []
    for node1,node2 in edges_to_remove:
        last_nodes.append(node1)
        last_nodes.append(node2)
    last_nodes = set(last_nodes) # Nodes in edges to be removed 
    edges_to_remove = [] # Reset and re-obtain the edges_to_remove
    graph, expansion_nodes, edges_to_remove, case4s, formula_fail_edges = formula_attributer(
            last_nodes, graph, trans_groups, expansion_nodes, edges_to_remove, case4s, formula_fail_edges)
    # Remove the edges to be removed
    edge_remover(edges_to_remove, graph)

    for node,met_form in nx.get_node_attributes(graph, 'Formula').items():
        if met_form != None:
            met_class = compound_classifier(met_form)
            if met_class == 'No Match':
                graph.nodes()[node]['Comp_Class'] = 'Unknown'
            elif met_class == 'Multiple Matches':
                graph.nodes()[node]['Comp_Class'] = 'Unknown'
            else:
                graph.nodes()[node]['Comp_Class'] = met_class
        
    return graph # Return the graph


def compound_classifier(form):
    """Returns the 'predicted'/most probable compound class following the stoichiometry rules presented in Rivas-Ubach et al., 2018.

       Paper: Rivas-Ubach, A., Liu, Y., Bianchi, T. S., Tolić, N., Jansson, C., & Paša-Tolić, L. (2018). Moving beyond the van Krevelen
    Diagram: A New Stoichiometric Approach for Compound Classification in Organisms. Analytical Chemistry, 90(10), 6152–6160. 
    https://doi.org/10.1021/acs.analchem.8b00529.

       form: Dictionary; formula in dictionary format. Example: {'C': 6, 'H': 12, 'O': 6}.

       returns: One of Five Classes ('Nucleotide', 'Protein', 'Lipid', 'Amino-Sugar' and 'Phytochemical'), 'No Match' (if no class 
    criteria matches the formula) or 'Multiple Matches' (if the formula fits the criteria of multiple classes).
    """
    # N/P stuff
    
    # Nucleotides first because double matches should be considered Nucleotides
    if form['N'] >= 2:
        if form['P'] >= 1:
            if form['S'] == 0:
                if 0.5 <= form['O']/form['C'] < 1.7:
                    if 1 < form['H']/form['C'] < 1.8:
                        if 0.2 <= form['N']/form['C'] <= 0.5:
                            if 0.1 <= form['P']/form['C'] < 0.35: # Nicotinamide ribotide doesn't fit by a tiny bit 0.091
                                if 0.6 <= form['N']/form['P'] <= 5:
                                    mass = getmass(form['C'],form['H'],form['O'],form['N'],
                                                        form['S'],form['P'],form['Cl'],form['F'])
                                    if 305 < mass < 523:
                                        return 'Nucleotide'
    
    met_class = []
    
    # Protein both constraints and Amino-Sugars
    if form['N'] >= 1:
        # Protein both constraints
        if form['P']/form['C'] < 0.17:
            # Protein Constraint 1
            if 0.12 < form['O']/form['C'] <= 0.6:
                if 0.9 < form['H']/form['C'] < 2.5:
                    if 0.126 <= form['N']/form['C'] <= 0.7: 
                        met_class.append('Protein')
            
            # Protein Constraint 2
            elif 0.6 < form['O']/form['C'] <= 1:
                if 1.2 < form['H']/form['C'] < 2.5:
                    if 0.2 < form['N']/form['C'] <= 0.7: 
                        met_class.append('Protein')
        
        # Amino-Sugars
        if form['O'] >= 3:
            if form['O']/form['C'] >= 0.61:
                if form['H']/form['C'] >= 1.45:
                    if 0.07 < form['N']/form['C'] <= 0.2:
                        if form['P']/form['C'] < 0.3:
                            if form['P'] > 0:
                                if form['N']/form['P'] <= 2:
                                    met_class.append('Amino-Sugar')
                            else:
                                met_class.append('Amino-Sugar')             
    
    # Lipids
    if form['O']/form['C'] <= 0.6:
        if form['H']/form['C'] >= 1.32:
            if form['N']/form['C'] <= 0.126:
                if form['P']/form['C'] < 0.35:
                    if form['P'] > 0:
                        if form['N']/form['P'] <= 5: 
                            met_class.append('Lipid')
                    else:
                        met_class.append('Lipid')
                        
    # Carbohydrates
    if form['N'] == 0:
        if form['O']/form['C'] >= 0.8:
            if 1.65 <= form['H']/form['C'] < 2.7:
                met_class.append('Carbohydrate')
                
    # Phytochemicals
    if form['O']/form['C'] <= 1.15:
        if form['H']/form['C'] < 1.32:
            if form['N']/form['C'] < 0.126:
                if form['P']/form['C'] <= 0.2:
                    if form['P'] > 0:
                        if form['N']/form['P'] <= 3: 
                            met_class.append('Phytochemical')
                    else:
                        met_class.append('Phytochemical')
    
    if len(met_class) == 0:
        return 'No Match'
    elif len(met_class) == 1:
        return met_class[0]
    else:
        return 'Multiple Matches'

