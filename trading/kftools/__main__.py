from kftools.arguments import parse_arguments

def main():
    args = parse_arguments()
    args.func(args)
  
main()
