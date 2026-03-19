import argparse
from plotting import (
    fig5,
    fig7a, fig7b, 
    fig8a, fig8b,
    fig9a, fig9b,
    fig10a, fig10b, fig10c, fig10d,
    fig11a, fig11b,
    fig12a, fig12b
)

ACTIONS = {
    "5": fig5,
    "7a": fig7a,
    "7b": fig7b,
    "8a": fig8a,
    "8b": fig8b,
    "9a": fig9a,
    "9b": fig9b,
    "10a": fig10a,
    "10b": fig10b,
    "10c": fig10c,
    "10d": fig10d,
    "11a": fig11a,
    "11b": fig11b,
}

def main():

    parser = argparse.ArgumentParser(prog='Save svg files of manuscript figures')
    parser.add_argument('action', help='Figure name, e.g. 7a|7b|8a|8b...')
    parser.add_argument('-data_folder', default='manuscript_data/', help='data folder', type=str)
    parser.add_argument('-save_folder', default='figures/', help='saved figures folder', type=str)

    args = parser.parse_args()
    
    action_func = ACTIONS.get(args.action)

    if action_func is None:
        raise ValueError(f"Unknown action: {args.action}")
    elif args.action == '5':
        fig5(args.save_folder)
    else:
        action_func(args.data_folder, args.save_folder)     

    return

if __name__ == '__main__':
    main()
    