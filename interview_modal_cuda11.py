from modal import Stub, Image, method, gpu, Secret, Mount
from huggingface_hub import snapshot_download
import time
import json
from jinja2 import Template
from interview_cuda import *

def download_model(name, info = {}, **kwargs):
    for k,v in kwargs.items(): info[k] = v
    with open("./_info.json",'w') as f:
        json.dump({"model_name": name, **info}, f)
    snapshot_download(name, **kwargs)

def download_codegen_2p5_7b_mono_model(): download_model("Salesforce/codegen25-7b-mono")
def download_codegen_2p5_7b_multi_model(): download_model("Salesforce/codegen25-7b-multi")
def download_codegen_2_1b_multi_model(): download_model("Salesforce/codegen2-1B")
def download_codegen_2_3p7b_multi_model(): download_model("Salesforce/codegen2-3_7B")
def download_codegen_2_7b_multi_model(): download_model("Salesforce/codegen2-7B")

def download_replit_code_instruct_3b_model(): download_model("sahil2801/replit-code-instruct-glaive", info={"eos_token_id": 1 })
def download_replit_code_v1_3b_model(): download_model("replit/replit-code-v1-3b", info={"generate_args": { "stop_seq": ["###"]}})
def download_replit_code_v1p5_3b_model(): download_model("replit/replit-code-v1_5-3b", info={"generate_args": { "stop_seq": ["#","//"]}})
def download_replit_codeinstruct_v2_3b_model(): download_model("teknium/Replit-v2-CodeInstruct-3B", info={"eos_token_id": 1 })

def download_falcon_instruct_7b_model(): download_model("tiiuae/falcon-7b-instruct", allow_patterns=["*.json","*.model","pytorch*.bin"])

def download_vicuna_1p1_7b_model(): download_model("lmsys/vicuna-7b-v1.1")
def download_vicuna_1p1_13b_model(): download_model("lmsys/vicuna-13b-v1.1")
def download_vicuna_1p3_7b_model(): download_model("lmsys/vicuna-7b-v1.3")
def download_vicuna_1p3_13b_model(): download_model("lmsys/vicuna-13b-v1.3")

def download_ajibwa_code_13b_model(): download_model('ajibawa-2023/Code-13B')
def download_ajibwa_code_33b_model(): download_model('ajibawa-2023/Code-33B')
def download_ajibwa_python_code_13b_model(): download_model('ajibawa-2023/Python-Code-13B')
def download_ajibwa_python_code_33b_model(): download_model('ajibawa-2023/Python-Code-33B')

def download_llama2_7b_model(): download_model("meta-llama/Llama-2-7b-hf", ignore_patterns=["*.bin"])
def download_llama2_chat_7b_model(): download_model("meta-llama/Llama-2-7b-chat-hf", ignore_patterns=["*.bin"])
def download_llama2_chat_13b_model(): download_model("meta-llama/Llama-2-13b-chat-hf", ignore_patterns=["*.bin"])
def download_llama2_chat_7b_awq_model(): download_model("abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq")
def download_orca2_13b_awq_model(): download_model('TheBloke/Orca-2-13B-AWQ')
def download_orca2_13b_model(): download_model('microsoft/Orca-2-13b', ignore_patterns=["*.safetensors"])

def download_dolphin_llama2_7b_model(): download_model('ehartford/dolphin-llama2-7b')
def download_wizardlm_uncencored_llama2_13b_model(): download_model('ehartford/WizardLM-1.0-Uncensored-Llama2-13b')
def download_nous_hermes_llama2_13b_model(): download_model('NousResearch/Nous-Hermes-Llama2-13b')
def download_llama2_coder_7b_model(): download_model('mrm8488/llama-2-coder-7b', info = { 'generate_args': { 'stop_seq': ["###"] } })

def download_wizardcoder_ct2_model(): download_model('michaelfeil/ct2fast-WizardCoder-15B-V1.0')
def download_mythomix_l2_13b_model(): download_model('Gryphe/MythoMix-L2-13b')
def download_huginn_1p2_13b_model(): download_model('The-Face-Of-Goonery/Huginn-13b-v1.2')
def download_orca_mini_1p3_7b_gptq_model(): download_model('TheBloke/orca_mini_v3_13B-GPTQ', revision='gptq-4bit-32g-actorder_True')
def download_orca_mini_13b_model(): download_model('psmathur/orca_mini_v3_13b')
def download_octogeex_model(): download_model('bigcode/octogeex')
def download_octocoder_model(): download_model('bigcode/octocoder')

def download_platypus2_model(): download_model('Open-Orca/OpenOrca-Platypus2-13B', info={"eos_token_id": 2 })
def download_losslessmegacoder_7b_model(): download_model('rombodawg/LosslessMegaCoder-llama2-7b-mini')
def download_losslessmegacoder_13b_model(): download_model('rombodawg/LosslessMegaCoder-llama2-13b-mini')
def download_decicoder_1b_model(): download_model('Deci/DeciCoder-1b')
def download_stablecode_completion_alpha_3b_model(): download_model('stabilityai/stablecode-completion-alpha-3b')
def download_stablelm_3b_model(): download_model('stabilityai/stablelm-3b-4e1t')

def download_refact_1b_model(): download_model('smallcloudai/Refact-1_6B-fim')
def download_evol_replit_v1_model(): download_model('nickrosh/Evol-Replit-v1')
def download_decilm_6b_model(): download_model('Deci/DeciLM-6b')
def download_skycode_model(): download_model('SkyWork/SkyCode')

def download_codellama_instruct_7b_model(): download_model('TheBloke/CodeLlama-7B-Instruct-fp16')
def download_codellama_instruct_13b_model(): download_model('TheBloke/CodeLlama-13B-Instruct-fp16')
def download_codellama_instruct_13b_ct2_model(): download_model('piratos/ct2fast-codellama-13b-instruct-hf')

def download_codellama_instruct_13b_exl2_4p6bit_model(): download_model('turboderp/CodeLlama-13B-instruct-exl2', revision ='4.65bpw')
def download_codellama_instruct_13b_exl2_3p5bit_model(): download_model('turboderp/CodeLlama-13B-instruct-exl2', revision ='3.5bpw')
def download_codellama_instruct_13b_exl2_2p7bit_model(): download_model('turboderp/CodeLlama-13B-instruct-exl2', revision = '2.7bpw')

def download_llama2_chat_70b_exl2_2p55_model(): download_model('turboderp/Llama2-70B-chat-exl2', revision = '2.55bpw')
def download_llama2_chat_70b_exl2_2p4_model(): download_model('turboderp/Llama2-70B-chat-exl2', revision = '2.4bpw')
def download_llama2_chat_70b_exl2_4p0_model(): download_model('turboderp/Llama2-70B-chat-exl2', revision = '4.0bpw')

def download_codellama_instruct_34b_exl2_4p0_model(): download_model('turboderp/CodeLlama-34B-instruct-exl2', revision = '4.0bpw')
def download_codellama_instruct_34b_exl2_3p0_model(): download_model('turboderp/CodeLlama-34B-instruct-exl2', revision = '3.0bpw')
def download_codebooga_34b_exl2_4p25_model(): download_model('oobabooga/CodeBooga-34B-v0.1-EXL2-4.250b')
def download_codebooga_34b_exl2_3p0_model(): download_model('LoneStriker/CodeBooga-34B-v0.1-3.0bpw-h6-exl2')
def download_codebooga_34b_exl2_4p0_model(): download_model('LoneStriker/CodeBooga-34B-v0.1-4.0bpw-h6-exl2')
def download_codebooga_34b_exl2_5p0_model(): download_model('LoneStriker/CodeBooga-34B-v0.1-5.0bpw-h6-exl2')
def download_speechless_codellama_34b_model(): download_model('TheBloke/speechless-codellama-34b-v2.0-AWQ')

def download_codellama_7b_model(): download_model('TheBloke/CodeLlama-7B-fp16', info = { 'generate_args': { 'stop_seq': ["\n#","\n//"] } })
def download_codellama_13b_model(): download_model('TheBloke/CodeLlama-13B-fp16', info = { 'generate_args': { 'stop_seq': ["\n#","\n//"] } })

def download_codellama_python_7b_model(): download_model('TheBloke/CodeLlama-7B-Python-fp16', info = { 'generate_args': { 'stop_seq': ["\n#","\n//"] } })
def download_codellama_python_13b_model(): download_model('TheBloke/CodeLlama-13B-Python-fp16', info = { 'generate_args': { 'stop_seq': ["\n#","\n//"] } })
def download_nous_hermes_code_13b_model(): download_model('Undi95/Nous-Hermes-13B-Code', info = { 'tokenizer': 'NousResearch/Nous-Hermes-Llama2-13b' })
def download_codellama_oasst_13b_model(): download_model('OpenAssistant/codellama-13b-oasst-sft-v10')
def download_codellama_phind_v2_model(): download_model('TheBloke/Phind-CodeLlama-34B-v2-AWQ', info = { 'big_model': True })
def download_codellama_phind_v2_gptq_model(): download_model('TheBloke/Phind-CodeLlama-34B-v2-GPTQ', revision='gptq-4bit-32g-actorder_True')
def download_codellama_phind_v2_fp16_model(): download_model('Phind/Phind-CodeLlama-34B-v2')
def download_speechless_llama2_model(): download_model('uukuguy/speechless-llama2-hermes-orca-platypus-wizardlm-13b')
def download_codellama_instruct_34b_fp16_model(): download_model('TheBloke/CodeLlama-34B-Instruct-fp16')
def download_wizardcoder_python_34b_fp16_model(): download_model('WizardLM/WizardCoder-Python-34B-V1.0')
def download_codebooga_34b_fp16_model(): download_model('oobabooga/CodeBooga-34B-v0.1')
def download_wizardcoder_33b_1p1_34b_fp16_model(): download_model('WizardLM/WizardCoder-33B-V1.1')

def download_mistral_instruct_model(): download_model('mistralai/Mistral-7B-Instruct-v0.1')
def download_mistral_instruct_0p2_model(): download_model('mistralai/Mistral-7B-Instruct-v0.2')
def download_mistral_base_model(): download_model('mistralai/Mistral-7B-v0.1')
def download_airoboros_m_7b_312_model(): download_model('jondurbin/airoboros-m-7b-3.1.2')
def download_airoboros_m_7b_312_awq_model(): download_model('TheBloke/Airoboros-M-7B-3.1.2-AWQ')
def download_openorca_mistral_7b_model(): download_model('Open-Orca/Mistral-7B-OpenOrca')
def download_speechless_code_mistral_7b_model(): download_model('uukuguy/speechless-code-mistral-7b-v1.0')
def download_codeshell_7b_model(): download_model('WisdomShell/CodeShell-7B', info = { 'generate_args': { 'stop_seq': ["\n#","\n//"] } })
def download_openhermes_mistral_7b_modal(): download_model('teknium/OpenHermes-2.5-Mistral-7B')
def download_mistral_7b_code_16k_model(): download_model('Nondzu/Mistral-7B-code-16k-qlora', ignore_patterns=["*.bin"])
def download_mistral_7b_code_16k_awq_model(): download_model('TheBloke/Mistral-7B-Code-16K-qlora-AWQ', info = { 'tokenizer_mode': 'slow' })
def download_mistral_7b_code_16k_gptq_model(): download_model('TheBloke/Mistral-7B-Code-16K-qlora-GPTQ', revision='gptq-4bit-32g-actorder_True')
def download_mixtral_intervitens_exl2_35bpw_model(): download_model('intervitens/Mixtral-8x7B-Instruct-v0.1-3.5bpw-h6-exl2-rpcal')
def download_mixtral_turboderp_exl2_35bpw_model(): download_model('turboderp/Mixtral-8x7B-instruct-exl2', revision='3.5bpw')
def download_mixtral_turboderp_exl2_30bpw_model(): download_model('turboderp/Mixtral-8x7B-instruct-exl2', revision='3.0bpw')
def download_mixtral_turboderp_exl2_27bpw_model(): download_model('turboderp/Mixtral-8x7B-instruct-exl2', revision='2.7bpw')
def download_mixtral_turboderp_exl2_25bpw_model(): download_model('turboderp/Mixtral-8x7B-instruct-exl2', revision='2.5bpw')
def download_mixtral_intervitens_exl2_35bpw_norpcal_model(): download_model('intervitens/Mixtral-8x7B-Instruct-v0.1-3.5bpw-h6-exl2')
def download_mixtral_instruct_gptq_model(): download_model('TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', revision='gptq-4bit-32g-actorder_True')
def download_mixtral_instruct_gptq_3bit_model(): download_model('TheBloke/Mixtral-8x7B-v0.1-GPTQ', revision='gptq-3bit-128g-actorder_True')
def download_dolphin_mixtral_exl2_4bpw_model(): download_model('LoneStriker/dolphin-2.5-mixtral-8x7b-4.0bpw-h6-exl2-2')
def download_dolphin_mistral_gptq_model(): download_model('TheBloke/dolphin-2.5-mixtral-8x7b-GPTQ')

def download_dolphin_mistral_2p7_model(): download_model('cognitivecomputations/dolphin-2.7-mixtral-8x7b')
def download_dolphin_mistral_2p7_exl2_3bpw_model(): download_model('LoneStriker/dolphin-2.7-mixtral-8x7b-3.0bpw-h6-exl2')
def download_dolphin_mistral_2p7_exl2_4bpw_model(): download_model('LoneStriker/dolphin-2.7-mixtral-8x7b-4.0bpw-h6-exl2')

def download_mixtral_hqq_model(): download_model('mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-2bit_g16_s128-HQQ')
def download_mixtral_hqq_moe_2bpw_model(): download_model('mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bit-HQQ')
def download_mixtralx2_model(): download_model('cloudyu/Mixtral_7Bx2_MoE')
def download_codeninja_1p0_model(): download_model('beowolx/CodeNinja-1.0-OpenChat-7B')
def download_openchat_1210_model(): download_model('openchat/openchat-3.5-1210')
def download_mixtral_exl2_2p4bpw_model(): download_model('LoneStriker/Mixtral-8x7B-Instruct-v0.1-2.4bpw-h6-exl2')
def download_mixtral_exl2_3p0bpw_model(): download_model('LoneStriker/Mixtral-8x7B-Instruct-v0.1-3.0bpw-h6-exl2')
def download_mixtral_instruct_awq_model(): download_model('TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ')

def download_xwin_lm_70b_firelzrd_exl2_model(): download_model('firelzrd/Xwin-LM-70B-V0.1-exl2', revision='4_5-bpw')
def download_xwin_lm_70b_matatonic_exl2_model(): download_model('matatonic/Xwin-LM-70B-V0.1-exl2-4.800b')
def download_xwin_lm_70b_lonestriker_exl2_model(): download_model('LoneStriker/Xwin-LM-70B-V0.1-4.65bpw-h6-exl2')
def download_xwin_lm_70b_lonestriker_2p4_exl2_model(): download_model('LoneStriker/Xwin-LM-70B-V0.1-2.4bpw-h6-exl2')
def download_xwin_lm_7b_v2_model(): download_model('Xwin-LM/Xwin-LM-7B-V0.2')
def download_xwin_lm_13b_v2_model(): download_model('Xwin-LM/Xwin-LM-13B-V0.2')

def download_deepseek_6p7_instruct_model(): download_model('deepseek-ai/deepseek-coder-6.7b-instruct', info={'eos_token_id':32021})
def download_deepseek_6p7_awq_instruct_model(): download_model('TheBloke/deepseek-coder-6.7B-instruct-AWQ', info={'eos_token_id':32021})
def download_deepseek_1p3_instruct_model(): download_model('deepseek-ai/deepseek-coder-1.3b-instruct', info={'eos_token_id':32021})
def download_deepseek_1p3_awq_instruct_model(): download_model('TheBloke/deepseek-coder-1.3b-instruct-AWQ', info={'eos_token_id':32021})
def download_deepseek_33_awq_instruct_model(): download_model('TheBloke/deepseek-coder-33B-instruct-AWQ', info={'eos_token_id':32021})

def download_magicoder_s_ds_6p7b_model(): download_model('ise-uiuc/Magicoder-S-DS-6.7B')
def download_magicoder_ds_6p7b_model(): download_model('ise-uiuc/Magicoder-DS-6.7B')
def download_magicoder_s_cl_7b_model(): download_model('ise-uiuc/Magicoder-S-CL-7B')
def download_magicoder_cl_7b_model(): download_model('ise-uiuc/Magicoder-CL-7B')

def download_llm360_crystalcoder_7b_model(): download_model('LLM360/CrystalCoder', info={'max_model_len': 2048, 'generate_args': { 'stop_seq': ["\n#","\n//"] } })

# NOTE: the stripedhyna arhitecture has two additional dependnencies
#    .pip_install("git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm")
#    .pip_install("flash-attn >= 2.0.0")
def download_togethercomputer_stripedhyena_nous_7b_model(): download_model('togethercomputer/StripedHyena-Nous-7B', ignore_patterns=["*.bin"])

def download_microsoft_phi2_model(): download_model('microsoft/phi-2')
def download_dolphin_2p6_phi2_model(): download_model('cognitivecomputations/dolphin-2_6-phi-2')

def download_nous_hermes_2_yi_model(): download_model('LoneStriker/Nous-Hermes-2-Yi-34B-3.0bpw-h6-exl2')
def download_nous_hermes_2_yi_4bpw_model(): download_model('LoneStriker/Nous-Hermes-2-Yi-34B-4.0bpw-h6-exl2')
def download_nous_hermes_2_yi_gptq_model(): download_model('TheBloke/Nous-Hermes-2-Yi-34B-GPTQ', info = {'eos_token_id': 7})
def download_nous_hermes_2_yi_awq_model(): download_model('TheBloke/Nous-Hermes-2-Yi-34B-AWQ', info = {'eos_token_id': 7})

def download_tinyllama_chat_model(): download_model('TinyLlama/TinyLlama-1.1B-Chat-v1.0', info = { 'max_model_len': 2048 })
def download_openllamav2_3b_model(): download_model('openlm-research/open_llama_3b_v2', info = { 'max_model_len': 2048, 'tokenizer_args': { 'use_fast': False }, 'generate_args': { 'stop_seq': ["\n#","\n//"] } })
def download_openllamav2_7b_model(): download_model('openlm-research/open_llama_7b_v2', info = { 'max_model_len': 2048, 'tokenizer_args': { 'use_fast': False }, 'generate_args': { 'stop_seq': ["\n#","\n//"] } })
def download_openllamav2_instruct_7b_model(): download_model('VMware/open-llama-7b-v2-open-instruct', info = { 'max_model_len': 2048, 'tokenizer_args': { 'use_fast': False }, 'generate_args': { 'stop_seq': ["\n#","\n//"] }})

def download_nous_hermes_2_solar_model(): download_model('NousResearch/Nous-Hermes-2-SOLAR-10.7B', info={"eos_token_id": 32000 })

def download_code_millenials_13b_model(): download_model('budecosystem/code-millenials-13b')
def download_code_millenials_34b_model(): download_model('budecosystem/code-millenials-34b')

image = (
    Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04",
                        setup_dockerfile_commands=["RUN apt-get update", "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential"])
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.1",
        "optimum==1.15.0",
        "tiktoken==0.5.2",
        "bitsandbytes==0.41.3",
        "accelerate==0.25.0",
        "einops==0.6.1",
        "sentencepiece==0.1.99",
        "hf-transfer~=0.1",
        "scipy==1.10.1",
        "pyarrow==11.0.0",
        "hf-hub-ctranslate2>=2.0.8",
        "ctranslate2>=3.16.0",
        "xformers==0.0.23.post1+cu118",
        "https://github.com/vllm-project/vllm/releases/download/v0.2.6/vllm-0.2.6+cu118-cp310-cp310-manylinux1_x86_64.whl",
        "https://github.com/turboderp/exllamav2/releases/download/v0.0.11/exllamav2-0.0.11+cu118-cp310-cp310-linux_x86_64.whl",
        index_url="https://download.pytorch.org/whl/cu118",
        extra_index_url="https://pypi.org/simple"
    )  
    .pip_install(
        "auto-gptq==0.6.0",
        extra_index_url="https://huggingface.github.io/autogptq-index/whl/cu118/"
    )
    .run_commands(
        "git clone https://github.com/turboderp/exllama /repositories/exllama && cd /repositories/exllama && git checkout 3b013cd53c7d413cf99ca04c7c28dd5c95117c0d"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMP_NUM_THREADS": "8"})
    .pip_install("git+https://github.com/mobiusml/hqq.git@0.1.1")
    ##### SELECT MODEL HERE ##############    
    .run_function(download_dolphin_mistral_2p7_exl2_3bpw_model, secret=Secret.from_name("my-huggingface-secret"))
    ######################################
)
stub = Stub(image=image)

##### SELECT RUNTIME HERE #############
#RUNTIME = "transformers"
#QUANT = QUANT_FP16
#RUNTIME = "ctranslate2"
#RUNTIME = "vllm"
#RUNTIME = "autogptq"
#RUNTIME = "exllama"
RUNTIME = "exllama2"
#RUNTIME = "hqq"
#######################################

##### SELECT GPU HERE #################
#gpu_request = gpu.T4(count=1)
gpu_request = gpu.A10G(count=1)
#gpu_request = gpu.A100(count=1, memory=80)
#######################################

@stub.cls(gpu=gpu_request, cpu=8, concurrency_limit=1, container_idle_timeout=300, secret=Secret.from_name("my-huggingface-secret"), mounts=[Mount.from_local_python_packages("interview_cuda")])
class ModalWrapper:
    def __enter__(self):
        self.info = json.load(open('./_info.json'))

        if RUNTIME == "transformers":
            self.wrapper = InterviewTransformers(self.info['model_name'], self.info, quant=QUANT)
        elif RUNTIME == "vllm":
            gpu_split = 2 if gpu_request.count == 2 else None
            print(gpu_split)
            self.wrapper = InterviewVLLM(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "autogptq":
            self.wrapper = InterviewAutoGPTQ(self.info['model_name'], self.info)
        elif RUNTIME == "hqq":
            self.wrapper = InterviewHQQ(self.info['model_name'], self.info)
        elif RUNTIME == "exllama":
            gpu_split = '17,24' if gpu_request.count == 2 else None
            self.wrapper = InterviewExllama(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "exllama2":
            #
            self.wrapper = InterviewExllama2(self.info['model_name'], self.info)
        elif RUNTIME == "awq":
            if self.info.get('big_model'):
                gpu_split = '0,1' if gpu_request.count == 2 else '0,cpu'
            else:
                gpu_split = None
            self.wrapper = InterviewAWQ(self.info['model_name'], self.info, gpu_split=gpu_split)
        elif RUNTIME == "ctranslate2":
            self.wrapper = InterviewCtranslate2(self.info['model_name'], self.info)
        else:
            raise Exception("Unknown RUNTIME")

        self.wrapper.load()

    @method()
    def generate(self, prompt, params):
        return self.wrapper.generate(prompt, params)

# For local testing, run `modal run -q interview_modal.py --input results/prepare.ndjson --params params/precise.json`
@stub.local_entrypoint()
def main(input: str, params: str, iterations: int = 1, templateout: str = "", batch: bool = False):
    from prepare import save_interview
    from interview_cuda import interview_run

    output_template = Template(open(templateout).read()) if templateout else None

    tasks = []
    for param_file in params.split(','):
        for input_file in input.split(','):
            if param_file != '' and input_file != '':
                tasks.append((param_file, input_file))

    model = ModalWrapper()

    for param_file, input_file in tasks:
      interview = [json.loads(line) for line in open(input_file)]
      params_json = json.load(open(param_file,'r'))

      for iter in range(iterations):
        print(f"Starting iteration {iter} of {param_file} {input_file}")
        results, remote_info = interview_run(RUNTIME, model.generate.remote, interview, params_json, output_template, batch=batch )
        save_interview(input_file, templateout if templateout else 'none', param_file, remote_info['model_name'], results)
