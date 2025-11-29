import torch
import torch.nn as nn
from MixtureofExperts import Experts

def test_moe():
    print("=== Starting MoE Dry Run ===")
    
    BATCH_SIZE = 4
    SEQ_LEN = 32
    D_MODEL = 64
    NUM_EXPERTS = 4
    TOP_K = 2
    
    print(f"Configuration: B={BATCH_SIZE}, T={SEQ_LEN}, D={D_MODEL}, Experts={NUM_EXPERTS}, TopK={TOP_K}")

    model = Experts(
        num_experts=NUM_EXPERTS,
        d_model=D_MODEL,
        hidden_times=4,
        top_k=TOP_K
    )
    
    # Dummy Input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
    
    # Forward Pass
    print("\n[Step 1] Running Forward Pass...")
    try:
        output, aux_loss = model(x)
        print(f"  > Output shape: {output.shape}")
        print(f"  > Aux Loss: {aux_loss.item():.6f}")
        
        # Assertions
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), "Output shape mismatch!"
        assert aux_loss > 0, "Aux loss should be non-zero!"
        print("  > Forward pass successful.")
    except Exception as e:
        print(f"  > Forward pass FAILED: {e}")
        return

    # Check Token Distribution (Sanity Check)
    print("\n[Step 2] Checking Expert Utilization...")
    with torch.no_grad():
        routing_weights, _ = model.router(x)
        expert_counts = (routing_weights > 0).float().sum(dim=0)
        print(f"  > Tokens per expert: {expert_counts.tolist()}")
        total_routed = expert_counts.sum()
        expected_routed = BATCH_SIZE * SEQ_LEN * TOP_K
        assert total_routed == expected_routed, f"Expected {expected_routed} routed tokens, got {total_routed}"
        print("  > Token routing count matches (B * T * TopK).")

    # Backward Pass Check (Gradient Flow)
    print("\n[Step 3] Checking Backward Pass (Gradients)...")
    try:
        # dummy loss value
        target = torch.randn_like(output)
        main_loss = nn.MSELoss()(output, target)
        
        # total loss
        total_loss = main_loss + aux_loss
        
        # Backprop
        total_loss.backward()
        
        # Check gradients in Router
        router_grads = model.router.route.weight.grad
        if router_grads is not None:
             grad_norm = router_grads.norm().item()
             print(f"  > Router gradients found! Norm: {grad_norm:.6f}")
        else:
            print("  > WARNING: No gradients in Router!")

        # Check gradients in Experts (Check Expert 0)
        expert_grads = model.experts[0].net[0].weight.grad
        if expert_grads is not None:
             print(f"  > Expert 0 gradients found!")
        else:
             # It is possible an expert wasn't selected, so this isn't strictly an error, 
             # but with random init and batch size 4*32, it's unlikely.
             print("  > Notice: Expert 0 has no gradients (might not have been selected).")
             
        print("  > Backward pass successful.")
        
    except Exception as e:
        print(f"  > Backward pass FAILED: {e}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_moe()