# By lllyasviel


import torch


cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    # Entry log
    print(f'[DETAIL LOG] Enter move_model_to_device_with_memory_preservation for {model.__class__.__name__}. Target: {target_device}. Preserved Mem: {preserved_memory_gb}GB. Current Free VRAM: {get_cuda_free_memory_gb(target_device):.2f}GB')
    # Original print statement (optional, kept for now)
    # print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    # This is the loop structure from the erroneous Turn 12 edit.
    # The prompt for Turn 15 asks to add logging to the *current* code.
    # The original code had an early exit in the loop. This version does not.
    # The logging will reflect the behavior of *this specific version* of the code.
    
    # The "Global VRAM check" logs from Turn 12 are removed to avoid redundancy with new specific logs.
    # The `modules_processed_count` and `m_idx` from Turn 12 are also removed to simplify to what Turn 15 asks.

    for m in model.modules():
        # Log module being processed
        print(f'[DETAIL LOG]  - Processing module: {m.__class__.__name__}')

        if hasattr(m, 'weight') and m.weight is not None:
            # Log current device of module's weight
            print(f'[DETAIL LOG]    - Module {m.__class__.__name__} weight device: {m.weight.device}')
            
            current_free_vram_before_move = get_cuda_free_memory_gb(target_device)
            # Inverted condition: move only if free VRAM is GREATER than preserved.
            if current_free_vram_before_move > preserved_memory_gb:
                print(f'[DETAIL LOG]    - Condition met (Free VRAM {current_free_vram_before_move:.2f}GB > Preserved {preserved_memory_gb}GB). Attempting to move {m.__class__.__name__} to {target_device}')
                try:
                    m.to(device=target_device)
                except Exception as e_module_move: # Can be torch.cuda.OutOfMemoryError if more specific catch is needed
                    print(f'[DETAIL LOG]    - ERROR moving module {m.__class__.__name__} to {target_device}: {e_module_move}')
                    print(f'[DETAIL LOG]    - Breaking loop due to error during module move.')
                    break # Stop further processing if a module fails to move
            else:
                # Condition NOT met (VRAM <= Preserved), so skip and break.
                print(f'[DETAIL LOG]    - Condition NOT met (Free VRAM {current_free_vram_before_move:.2f}GB <= Preserved {preserved_memory_gb}GB). Skipping move for {m.__class__.__name__} and stopping further GPU loading.')
                break # Stop processing further modules
        else:
            # Log non-parameter module
            print(f'[DETAIL LOG]    - Module {m.__class__.__name__} has no weight attribute or weight is None, skipping direct move.')

    # Log after loop. The aggressive model.to() is removed.
    print(f'[DETAIL LOG]  Finished module iteration. Current Free VRAM: {get_cuda_free_memory_gb(target_device):.2f}GB.')
    # model.to(device=target_device) # This line is REMOVED as per instructions
    torch.cuda.empty_cache()
    # Exit log
    print(f'[DETAIL LOG] Exit move_model_to_device_with_memory_preservation for {model.__class__.__name__}. Final Free VRAM: {get_cuda_free_memory_gb(target_device):.2f}GB')
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    initial_free_vram = get_cuda_free_memory_gb(target_device)
    print(f'[DETAIL OFFLOAD LOG] Enter offload_model_from_device_for_memory_preservation for {model.__class__.__name__}. Target device for offload checks: {target_device}. Preserved Mem Target: {preserved_memory_gb}GB. Initial Free VRAM: {initial_free_vram:.2f}GB')
    # Original print, can be kept or removed.
    # print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    if initial_free_vram >= preserved_memory_gb:
        print(f'[DETAIL OFFLOAD LOG] Initial VRAM ({initial_free_vram:.2f}GB) already meets or exceeds target ({preserved_memory_gb}GB). No offload needed. Exiting early.')
        torch.cuda.empty_cache()
        print(f'[DETAIL OFFLOAD LOG] Exit offload_model_from_device_for_memory_preservation for {model.__class__.__name__}. Final Free VRAM: {get_cuda_free_memory_gb(target_device):.2f}GB')
        return

    for m_idx, m in enumerate(model.modules()):
        print(f'[DETAIL OFFLOAD LOG]  - Processing module ({m_idx+1}): {m.__class__.__name__}')

        if hasattr(m, 'weight') and m.weight is not None:
            print(f'[DETAIL OFFLOAD LOG]    - Module {m.__class__.__name__} weight device: {m.weight.device}')
            if m.weight.device == target_device:
                current_free_gb = get_cuda_free_memory_gb(target_device)
                if current_free_gb < preserved_memory_gb:
                    print(f'[DETAIL OFFLOAD LOG]      - Condition met (Free VRAM {current_free_gb:.2f}GB < Preserved {preserved_memory_gb}GB). Offloading {m.__class__.__name__} from {target_device} to cpu.')
                    try:
                        m.to(device=cpu)
                    except Exception as e_offload_module:
                        print(f'[DETAIL OFFLOAD LOG]      - ERROR offloading module {m.__class__.__name__}: {e_offload_module}')
                else:
                    print(f'[DETAIL OFFLOAD LOG]      - Condition NOT met (Free VRAM {current_free_gb:.2f}GB >= Preserved {preserved_memory_gb}GB). Module {m.__class__.__name__} remains on {target_device}. Stopping further module-level offload attempts as target met.')
                    # If the target is met, we should stop trying to offload more modules one by one.
                    # The main function-level check at the start handles cases where VRAM is already sufficient.
                    # This break ensures we don't continue looping if partial offload meets the target.
                    break 
            else:
                print(f'[DETAIL OFFLOAD LOG]    - Module {m.__class__.__name__} is not on {target_device} (actual: {m.weight.device}), skipping offload.')
        else:
            print(f'[DETAIL OFFLOAD LOG]    - Module {m.__class__.__name__} has no weight attribute or weight is None, skipping direct offload.')
        
        # Check VRAM again after potential offload; if target met, exit loop.
        # This is similar to the original function's loop-level check.
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            print(f'[DETAIL OFFLOAD LOG]  VRAM target ({preserved_memory_gb}GB) met after processing module {m.__class__.__name__}. Current Free VRAM: {get_cuda_free_memory_gb(target_device):.2f}GB. Stopping module loop.')
            break

    # This part is reached if the loop completes and VRAM target might still not be met.
    final_check_free_vram = get_cuda_free_memory_gb(target_device)
    if final_check_free_vram < preserved_memory_gb:
        print(f'[DETAIL OFFLOAD LOG]  Finished module iteration. Free VRAM ({final_check_free_vram:.2f}GB) still below target ({preserved_memory_gb}GB). Attempting full model.to(cpu).')
        try:
            model.to(device=cpu)
        except Exception as e_offload_full:
            print(f'[DETAIL OFFLOAD LOG]  ERROR during full model.to(cpu): {e_offload_full}')
    else:
        print(f'[DETAIL OFFLOAD LOG]  Finished module iteration. Free VRAM ({final_check_free_vram:.2f}GB) meets or exceeds target ({preserved_memory_gb}GB). Full model.to(cpu) not needed.')
        
    torch.cuda.empty_cache()
    print(f'[DETAIL OFFLOAD LOG] Exit offload_model_from_device_for_memory_preservation for {model.__class__.__name__}. Final Free VRAM: {get_cuda_free_memory_gb(target_device):.2f}GB')
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
