"""Text inference for the complete ReSI-Safin-1-35B-A3B model."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='AI45Research/ReSI-Safin-1-35B-A3B',
                        help='Hugging Face model ID or downloaded model directory')
    parser.add_argument('--prompt', default='Explain why the sky is blue in three sentences.')
    parser.add_argument('--system', help='Optional system message')
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--temperature', type=float, help='Override the saved sampling temperature')
    parser.add_argument('--top-p', type=float, help='Override the saved nucleus sampling probability')
    parser.add_argument('--top-k', type=int, help='Override the saved top-k setting')
    parser.add_argument('--greedy', action='store_true', help='Disable sampling')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.temperature is not None and args.temperature <= 0:
        parser.error('Use --greedy for deterministic decoding; temperature must be positive')
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error('--top-p must be in (0, 1]')
    if args.top_k is not None and args.top_k < 0:
        parser.error('--top-k must be nonnegative')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device_map,
        trust_remote_code=True, attn_implementation='eager',
    ).eval()
    messages = []
    if args.system:
        messages.append({'role': 'system', 'content': args.system})
    messages.append({'role': 'user', 'content': args.prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(
        model.get_input_embeddings().weight.device)
    settings = {'max_new_tokens': args.max_new_tokens}
    for name in ('temperature', 'top_p', 'top_k'):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if args.greedy:
        settings.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        output = model.generate(
            **inputs, **settings,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
        )
    generated = output[0, inputs.input_ids.shape[-1]:]
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else eos)
    if len(generated) >= args.max_new_tokens and int(generated[-1]) not in eos_ids:
        print('Output reached --max-new-tokens; increase it if the answer is incomplete.', file=sys.stderr)


if __name__ == '__main__':
    main()
