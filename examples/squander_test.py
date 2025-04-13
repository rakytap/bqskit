from bqskit.ir import Circuit





circuit_name = 'tutorials/heisenberg-16-20' #9symml_195'


bqskit_circuit_original = Circuit.from_file( circuit_name + '.qasm')


from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner, ClusteringPartitioner, GreedyPartitioner, ScanPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass, LEAPSynthesisPass, QFASTDecompositionPass, QPredictDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.passes import SquanderSynthesisPass
from bqskit.qis.unitary import Unitary
from qiskit import transpile
import numpy as np
import time
import pickle 
# import


# define the largest partition in circuit
largest_partition = 3


print("\n the original gates are: \n")

    
original_gates = []

for gate in bqskit_circuit_original.gate_set:
    case_original = {f"{gate}count:": bqskit_circuit_original.count(gate)}
    original_gates.append(case_original)
    
print(original_gates, "\n")


###########################################################################
# SQUANDER Tree seach synthesis 
 
start_squander = time.time()
 
config = {  'strategy': "Tree_search", 
            'parallel': 0,
         }
                

workflow = [
    QuickPartitioner( largest_partition ), 
    ForEachBlockPass([SquanderSynthesisPass(squander_config=config, optimizer_engine="BFGS" ), ScanningGateRemovalPass()]), 
    UnfoldPass(),
]



# Finally, we construct a compiler and submit the task
#with Compiler(num_workers=1) as compiler:
with Compiler() as compiler:
    circuit_squander_tree = compiler.compile(bqskit_circuit_original, workflow)


Circuit.save(circuit_squander_tree, circuit_name + '_squander_tree_search.qasm')




print("\n Circuit optimized with squander tree search:")


squander_gates = []

for gate in circuit_squander_tree.gate_set:
    case_squander = {f"{gate}count:":  circuit_squander_tree.count(gate)}
    squander_gates.append(case_squander)
 
end_squander = time.time()
time_squander = "the execution time with squander tree search:" + str(end_squander-start_squander)

print(squander_gates, "\n")
print( time_squander )
print(' ')
print(' ')
 

    
    

###########################################################################
# SQUANDER Tabu seach synthesis 
 
start_squander = time.time()

config = {  'strategy': "Tabu_search", 
            'parallel': 0,
         }

workflow = [
    QuickPartitioner( largest_partition ), 
    ForEachBlockPass([SquanderSynthesisPass(squander_config=config, optimizer_engine="BFGS" ), ScanningGateRemovalPass()]), 
    UnfoldPass(),
]


# Finally, we construct a compiler and submit the task
#with Compiler(num_workers=1) as compiler:
with Compiler() as compiler:
    circuit_squander_tabu = compiler.compile(bqskit_circuit_original, workflow)


Circuit.save(circuit_squander_tabu, circuit_name + '_squander_tabu_search.qasm')




print("\n Circuit optimized with squander tabu search:")


squander_gates = []

for gate in circuit_squander_tabu.gate_set:
    case_squander = {f"{gate}count:":  circuit_squander_tabu.count(gate)}
    squander_gates.append(case_squander)
 
end_squander = time.time()
time_squander = "the execution time with squander with tabu search:" + str(end_squander-start_squander)

print(squander_gates, "\n")
print( time_squander )
print(' ')
print(' ')
  





###########################################################################
# QSearch synthesis

start_qsearch = time.time()


workflow = [
    QuickPartitioner( largest_partition ), 
    ForEachBlockPass([QSearchSynthesisPass(), ScanningGateRemovalPass()]), 
    UnfoldPass(),
]

# Finally, we construct a compiler and submit the task
with Compiler() as compiler:
    synthesized_circuit_qsearch = compiler.compile(bqskit_circuit_original, workflow)


# save the circuit is qasm format
Circuit.save(synthesized_circuit_qsearch, circuit_name + '_qsearch.qasm')



print("\n Circuit optimized with qsearch:")

qsearch_gates = []

for gate in synthesized_circuit_qsearch.gate_set:
    case_qsearch = {f"{gate}count:":  synthesized_circuit_qsearch.count(gate)}   
    qsearch_gates.append(case_qsearch)
 
end_qsearch = time.time()
time_qsearch = "the execution time with qsearch:" + str(end_qsearch-start_qsearch)

print(qsearch_gates, "\n")
print( time_qsearch )
print(' ')
print(' ')




##############################################################################
#################### Test the generated circuits #############################

import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit
import qiskit_aer as Aer    
   

if qiskit_version[0] == '0':
    from qiskit import execute
else:
    from qiskit import transpile



# load the circuit from QASM format
qc_original      = QuantumCircuit.from_qasm_file( circuit_name +  '.qasm' )
qc_squander_tabu = QuantumCircuit.from_qasm_file( circuit_name +  '_squander_tabu_search.qasm' )
qc_squander_tree = QuantumCircuit.from_qasm_file( circuit_name +  '_squander_tree_search.qasm' )
qc_qsearch       = QuantumCircuit.from_qasm_file( circuit_name +  '_qsearch.qasm' )


# generate random initial state on which we test the circuits
matrix_size = 1 << qc_original.num_qubits
initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state = initial_state_real + initial_state_imag*1j
initial_state = initial_state/np.linalg.norm(initial_state)


state_to_transform = initial_state.copy()
qc_original.initialize( state_to_transform )

state_to_transform = initial_state.copy()
qc_squander_tabu.initialize( state_to_transform )

state_to_transform = initial_state.copy()
qc_squander_tree.initialize( state_to_transform )

state_to_transform = initial_state.copy()
qc_qsearch.initialize( state_to_transform )



        
if qiskit_version[0] == '0':
	
	# Select the StatevectorSimulator from the Aer provider
	simulator = Aer.get_backend('statevector_simulator')	
		
	backend = Aer.get_backend('aer_simulator')

	# Execute and get the state vector
	result                          = execute(qc_original, simulator).result()
	transformed_state_original      = np.array( result.get_statevector(qc_original) )

	result                          = execute(qc_squander_tabu, simulator).result()
	transformed_state_squander_tabu = np.array( result.get_statevector(qc_squander_tabu) )

	result                          = execute(qc_squander_tree, simulator).result()
	transformed_state_squander_tree = np.array( result.get_statevector(qc_squander_tree) )
	
	result                          = execute(qc_qsearch, simulator).result()
	transformed_state_qsearch       = np.array( result.get_statevector(qc_qsearch) )


# Execute and get the state vector
else :
	
	qc_original.save_statevector()
	qc_squander_tabu.save_statevector()
	qc_squander_tree.save_statevector()
	qc_qsearch.save_statevector()

	backend = Aer.AerSimulator(method='statevector')

	# Execute and get the state vector
	result                          = backend.run(qc_original).result()
	transformed_state_original      = np.array( result.get_statevector(qc_original) )

	result                          = backend.run(qc_squander_tabu).result()
	transformed_state_squander_tabu = np.array( result.get_statevector(qc_squander_tabu) )

	result                          = backend.run(qc_squander_tree).result()
	transformed_state_squander_tree = np.array( result.get_statevector(qc_squander_tree) )
	
	result                          = backend.run(qc_qsearch).result()
	transformed_state_qsearch       = np.array( result.get_statevector(qc_qsearch) )

	
       
overlap_squander_tree = transformed_state_original.transpose().conjugate() @ transformed_state_squander_tree
overlap_squander_tree = overlap_squander_tree * overlap_squander_tree.conj()


overlap_squander_tabu = transformed_state_original.transpose().conjugate() @ transformed_state_squander_tabu
overlap_squander_tabu = overlap_squander_tabu * overlap_squander_tabu.conj()


overlap_qsearch = transformed_state_original.transpose().conjugate() @ transformed_state_qsearch
overlap_qsearch = overlap_qsearch * overlap_qsearch.conj()

print(' ')
print( 'The overlap of states obtained with the original and the squander compressed circuit with tree search: ',  overlap_squander_tree )
print( 'The overlap of states obtained with the original and the squander compressed circuit with tabu search: ',  overlap_squander_tabu )
print( 'The overlap of states obtained with the original and the qsearch compressed circuit: ',  overlap_qsearch )

