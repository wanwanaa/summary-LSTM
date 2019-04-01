from pyrouge import Rouge155

r = Rouge155()
r.system_dir = 'result/gold'
r.model_dir = 'result/data'
r.system_filename_pattern = 'some.(\d+).txt'
r.model_filename_pattern = 'some.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)